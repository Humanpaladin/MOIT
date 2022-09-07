import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from transformers import BertTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(
            input_ids=src_tokens,
            attention_mask=mask,
            return_dict=True,
            output_hidden_states=True
        )
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states

        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1]+1
        # 这里在pipe中设置了第0个位置是sos, 第一个位置是eos, 所以做一下映射
        mapping = torch.LongTensor([0, 2]+sorted(label_ids, reverse=False))
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask
        first = state.first
        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]
        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values
        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)
        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return self(tokens, state)[:, -1]


class CaGFBartDecoder(FBartDecoder):
    def __init__(
            self,
            decoder,
            pad_token_id,
            label_ids,
            avg_feature=True,
            use_encoder_mlp=False
    ):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature
        hidden_size = decoder.embed_tokens.weight.size(1)

    def forward(self, gold_index, state):
        # bsz, max_len = gold_index.size()
        encoder_outputs = state.encoder_output      # tensor(2, 28, 768)
        encoder_pad_mask = state.encoder_mask       # tensor(2, 28)
        first = state.first
        cumsum = gold_index.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = gold_index.lt(self.src_start_index)
        mapped_tokens = gold_index.masked_fill(gold_index.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]
        src_tokens_index_temp = gold_index - self.src_start_index
        gold_index_without_special_tokens = src_tokens_index_temp.masked_fill(src_tokens_index_temp.lt(0), 0)
        src_token_ids = state.src_tokens
        if first is not None:
            # src_tokens = src_token_ids.gather(index=first, dim=1)
            src_token_ids = src_token_ids.gather(index=first, dim=1)

        word_mapped_tokens = src_token_ids.gather(index=gold_index_without_special_tokens, dim=1)
        token_ids = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        token_ids = token_ids.masked_fill(tgt_pad_mask, self.pad_token_id)
        if self.training:
            token_ids = token_ids[:, :-1]
            decoder_pad_mask = token_ids.eq(self.pad_token_id)  # decoder需要让pad位置为 1
            dict = self.decoder(
                input_ids=token_ids,
                decoder_padding_mask=decoder_pad_mask,
                encoder_hidden_states=encoder_outputs,
                encoder_padding_mask=encoder_pad_mask,
                decoder_causal_mask=self.causal_masks[:token_ids.size(1), :token_ids.size(1)],
                return_dict=True
            )
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=token_ids,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                use_pos_cache=token_ids.size(1)>3)

        hidden_state = dict.last_hidden_state

        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full(
            (hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_token_ids.size(-1)),
            fill_value=-1e24
        )

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])

        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:       # first == None
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(src_token_ids)

        if self.avg_feature:                                    # False
            src_outputs = (src_outputs + input_embed)/2

        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)

        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)
            word_scores = (gen_scores + word_scores)/2

        mask = mask.__or__(src_token_ids.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits


class CopyCaGFBartDecoder(FBartDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=True):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(hidden_size, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, tokens, state):
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  # 已经映射之后的分数

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class


        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(src_tokens)  # bsz x max_word_len x hidden_size
        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            gate = self.gate(hidden_state)  # bsz x max_len x 1
            word_scores = (gate*gen_scores + (1-gate)*word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits


class Restricter(nn.Module):
    def __init__(self, label_ids):
        super().__init__()
        self.src_start_index = 2+len(label_ids)
        self.tag_tokens = label_ids

    def __call__(self, state, tokens, scores, num_beams=1):
        """

        :param state: 各种DecoderState
        :param tokens: bsz x max_len，基于这些token，生成了scores的分数
        :param scores: bsz x vocab_size*real_num_beams  # 已经log_softmax后的
        :param int num_beams: 返回的分数和token的shape
        :return:
        num_beams==1:
            scores: bsz x 1, tokens: bsz x 1
        num_beams>1:
            scores: bsz x num_beams, tokens: bsz x num_beams  # 其中scores是从大到小排好序的
        """
        bsz, max_len = tokens.size()
        logits = scores.clone()
        if max_len > 1 and max_len % 5 == 1:  # 只能是tags
            logits[:, :2].fill_(-1e24)
            logits[:, self.src_start_index:].fill_(-1e24)
        elif max_len % 5 == 2:
            logits[:, 2:self.src_start_index].fill_(-1e24)
        else:
            logits[:, :self.src_start_index].fill_(-1e24)

        _, ids = torch.topk(logits, num_beams, dim=1, largest=True, sorted=True)  # (bsz, num_beams)
        next_scores = scores.gather(index=ids, dim=1)

        return next_scores, ids


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(
            cls,
            bart_model,                 # bart_model: 'facebook/bart-base'
            tokenizer,                  # tokenizer: BartTokenizer: 50266
            label_ids,                  # label_ids: [50265] 应为 label_id 的个数
            decoder_type=None,          # decoder_type: 'avg_score'
            copy_gate=False,            # copy_gate: False
            use_encoder_mlp=False,      # use_encoder_mlp: 1
            use_recur_pos=False,        # use_recur_pos: False
            tag_first=False             # tag_first: False
    ):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape                             # num_tokens: 50265
                                                                                            # _: 768

        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)     # tokenizer.unique_no_split_tokens: ['<s>', '</s>', '<<tag>>', '<mask>', '<pad>', '<unk>', '<<tag>>']
                                                                                            # num_tokens: 50265
                                                                                            # model.resize_token_embeddings(..) 使得
                                                                                            #   model.vocab 由 50265 -> 50272
        encoder = model.encoder         # encoder: BartEncoder
        decoder = model.decoder         # decoder: BartDecoder

        if use_recur_pos:               # use_recur_pos: False
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)

        for token in tokenizer.unique_no_split_tokens:                                      # tokenizer: BartTokenizer: 50266
                                                                                            #   为加了新 token {'<<tag>>': 50265} 的 tokenizer
                                                                                            #   tokenizer.unique_no_split_tokens:
                                                                                            #       ['</s>', '<<tag>>', '<mask>', '<pad>', '<s>', '<unk>', '<<tag>>']

            if token[:2] == '<<':  # 处理 '<<tag>>'
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))                  # '<<tag>>' 在 tokenizer 中的索引: [50265]
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]                                                                # '<<tag>>' 在词表中的索引 50265
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)

        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type == 'avg_score':
            if copy_gate:                   # copy_gate: False
                decoder = CopyCaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                              avg_feature=False, use_encoder_mlp=use_encoder_mlp)
            else:
                decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                          avg_feature=False, use_encoder_mlp=use_encoder_mlp)
        elif decoder_type == 'avg_feature':
            if copy_gate:
                decoder = CopyCaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                              avg_feature=True, use_encoder_mlp=use_encoder_mlp)
            else:
                decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                          avg_feature=True, use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")

        print(cls(encoder=encoder, decoder=decoder))

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")


class BartState(State):
    def __init__(
            self,
            encoder_output,             # 即为 encoder 最后一层的隐状态，为 tensor(2, 37, 768)
            encoder_mask,               # 即用 True/False 表示相应位置上是否为 <pad>，为 tensor(2, 37)
            src_tokens,                 # 即为输入句子的 token id，为 tensor(2, 37)
            first,
            src_embed_outputs           # 即为 encoder 返回的 hidden_states 中的 initial embedding，为 tensor(2, 37, 768)
    ):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new