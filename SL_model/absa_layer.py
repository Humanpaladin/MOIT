import logging
import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel
from torch.nn.modules.loss import CrossEntropyLoss
from seq_utils import labels_2_b_e_a_triplets, logsumexp, viterbi_decode
from bert import BertPreTrainedModel, XLNetPreTrainedModel


class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.3
        self.hidden_size = 768
        self.n_rnn_layers = 1       # not used if tagger is non-RNN model
        self.bidirectional = True   # not used if tagger is non-RNN model


class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels
        self.tagger_config = TaggerConfig()
        self.tagger_config.absa_type = bert_config.absa_type.lower()

        if bert_config.tfm_mode == 'finetune':
            self.bert = BertModel(bert_config)
        else:
            raise Exception("Invalid transformer mode %s !" % bert_config.tfm_mode)

        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        if bert_config.fix_tfm:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.tagger = None
        if self.tagger_config.absa_type == 'linear':
            penultimate_hidden_size = bert_config.hidden_size
        else:
            self.tagger_dropout = nn.Dropout(self.tagger_config.hidden_dropout_prob)
            if self.tagger_config.absa_type == 'lstm':
                pass
            elif self.tagger_config.absa_type == 'gru':
                pass
            elif self.tagger_config.absa_type == 'tfm':
                self.tagger = nn.TransformerEncoderLayer(
                    d_model=bert_config.hidden_size,
                    nhead=12,
                    dim_feedforward=4*bert_config.hidden_size,
                    dropout=0.1
                )
            elif self.tagger_config.absa_type == 'san':
                pass
            elif self.tagger_config.absa_type == 'crf':
                pass
            else:
                raise Exception("Unimplemented tagger type %s !" % self.tagger_config.absa_type)
            penultimate_hidden_size = self.tagger_config.hidden_size
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, input_masks=None, segment_ids=None, label_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        tagger_input = outputs[0]
        tagger_input = self.bert_dropout(tagger_input)

        if self.tagger is None or self.tagger_config.absa_type == 'crf':
            logits = self.classifier(tagger_input)
        else:
            if self.tagger_config.absa_type == 'lstm':
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'gru':
                classifier_input, _ = self.tagger(tagger_input)
            elif self.tagger_config.absa_type == 'san' or self.tagger_config.absa_type == 'tfm':
                tagger_input = tagger_input.transpose(0, 1)
                classifier_input = self.tagger(tagger_input)
                classifier_input = classifier_input.transpose(0, 1)
            else:
                raise Exception("Unimplemented downstream tagger layer %s ..." % self.tagger_config.absa_type)
            classifier_input = self.tagger_dropout(classifier_input)

            logits = self.classifier(classifier_input)
        outputs = (logits,) + outputs[2:]
        if label_ids is not None:
            if self.tagger_config.absa_type != 'crf':
                loss_func = CrossEntropyLoss()
                if input_masks is not None:
                    effective_token_flag = input_masks.view(-1) == 1
                    effective_token_logits = logits.view(-1, self.num_labels)[effective_token_flag]
                    effective_token_label_ids = label_ids.view(-1)[effective_token_flag]
                    batch_loss = loss_func(effective_token_logits, effective_token_label_ids)
                else:
                    batch_loss = loss_func(logits.view(-1, self.num_labels), label_ids.view(-1))

                outputs = (batch_loss,) + outputs
            else:
                log_likelihood = self.tagger(inputs=logits, tags=label_ids, mask=input_masks)
                loss = -log_likelihood
                outputs = (loss,) + outputs

        return outputs


class LSTM(nn.Module):
    # customized LSTM with layer normalization
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """
        :param input_size:
        :param hidden_size:
        :param bidirectional:
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.LNx = nn.LayerNorm(4*self.hidden_size)
        self.LNh = nn.LayerNorm(4*self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size, out_features=4*self.hidden_size, bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size, out_features=4*self.hidden_size, bias=True)

    def forward(self, x):
        """
        :param x: input, shape: (batch_size, seq_len, input_size)
        :return:
        """
        def recurrence(xt, hidden):
            """
            recurrence function enhanced with layer norm
            :param input: input to the current cell
            :param hidden:
            :return:
            """
            htm1, ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft * ctm1) + (it * gt)
            ht = ot * torch.tanh(self.LNc(ct))  # n_b x hidden_dim

            return ht, ct
        output = []
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        input = x.transpose(0, 1)
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden[0])
        # (bs, seq_len, hidden_size)
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        c_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0, c_0


class GRU(nn.Module):
    # customized GRU with layer normalization
    def __init__(self, input_size, hidden_size, bidirectional=True):
        """

        :param input_size:
        :param hidden_size:
        :param bidirectional:
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.Wxrz = nn.Linear(in_features=self.input_size, out_features=2*self.hidden_size, bias=True)
        self.Whrz = nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size, bias=True)
        self.Wxn = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=True)
        self.Whn = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=True)
        self.LNx1 = nn.LayerNorm(2*self.hidden_size)
        self.LNh1 = nn.LayerNorm(2*self.hidden_size)
        self.LNx2 = nn.LayerNorm(self.hidden_size)
        self.LNh2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        """

        :param x: input tensor, shape: (batch_size, seq_len, input_size)
        :return:
        """
        def recurrence(xt, htm1):
            """

            :param xt: current input
            :param htm1: previous hidden state
            :return:
            """
            gates_rz = torch.sigmoid(self.LNx1(self.Wxrz(xt)) + self.LNh1(self.Whrz(htm1)))
            rt, zt = gates_rz.chunk(2, 1)
            nt = torch.tanh(self.LNx2(self.Wxn(xt))+rt*self.LNh2(self.Whn(htm1)))
            ht = (1.0-zt) * nt + zt * htm1
            return ht

        steps = range(x.size(1))
        bs = x.size(0)
        hidden = self.init_hidden(bs)
        # shape: (seq_len, bsz, input_size)
        input = x.transpose(0, 1)
        output = []
        for t in steps:
            hidden = recurrence(input[t], hidden)
            output.append(hidden)
        # shape: (bsz, seq_len, input_size)
        output = torch.stack(output, 0).transpose(0, 1)

        if self.bidirectional:
            output_b = []
            hidden_b = self.init_hidden(bs)
            for t in steps[::-1]:
                hidden_b = recurrence(input[t], hidden_b)
                output_b.append(hidden_b)
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output, None

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0


class SAN(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super(SAN, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """

        :param src:
        :param src_mask:
        :param src_key_padding_mask:
        :return:
        """
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        # apply layer normalization
        src = self.norm(src)
        return src


class CRF(nn.Module):
    # borrow the code from
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    def __init__(self, num_tags, constraints=None, include_start_end_transitions=None):
        """

        :param num_tags:
        :param constraints:
        :param include_start_end_transitions:
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.include_start_end_transitions = include_start_end_transitions
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint_mask = torch.Tensor(self.num_tags+2, self.num_tags+2).fill_(1.)
        if include_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        # register the constraint_mask
        self.constraint_mask = nn.Parameter(constraint_mask, requires_grad=False)
        self.reset_parameters()

    def forward(self, inputs, tags, mask=None):
        """

        :param inputs: (bsz, seq_len, num_tags), logits calculated from a linear layer
        :param tags: (bsz, seq_len)
        :param mask: (bsz, seq_len), mask for the padding token
        :return:
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def reset_parameters(self):
        """
        initialize the parameters in CRF
        :return:
        """
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            nn.init.normal_(self.start_transitions)
            nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits, mask):
        """

        :param logits: emission score calculated by a linear layer, shape: (batch_size, seq_len, num_tags)
        :param mask:
        :return:
        """
        bsz, seq_len, num_tags = logits.size()
        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for t in range(1, seq_len):
            # iteration starts from 1
            emit_scores = logits[t].view(bsz, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(bsz, num_tags, 1)

            # calculate the likelihood
            inner = broadcast_alpha + emit_scores + transition_scores

            # mask the padded token when met the padded token, retain the previous alpha
            alpha = (logsumexp(inner, 1) * mask[t].view(bsz, 1) + alpha * (1 - mask[t]).view(bsz, 1))
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self, logits, tags, mask):
        """
        calculate the likelihood for the input tag sequence
        :param logits:
        :param tags: shape: (bsz, seq_len)
        :param mask: shape: (bsz, seq_len)
        :return:
        """
        bsz, seq_len, _ = logits.size()

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for t in range(seq_len-1):
            current_tag, next_tag = tags[t], tags[t+1]
            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[t].gather(1, current_tag.view(bsz, 1)).squeeze(1)

            score = score + transition_score * mask[t+1] + emit_score * mask[t]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, bsz)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def viterbi_tags(self, logits, mask):
        """

        :param logits: (bsz, seq_len, num_tags), emission scores
        :param mask:
        :return:
        """
        _, max_seq_len, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self.constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self.constraint_mask[:num_tags, :num_tags])
        )

        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                    self.start_transitions.detach() * self.constraint_mask[start_tag, :num_tags].data +
                    -10000.0 * (1 - self.constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                    self.end_transitions.detach() * self.constraint_mask[:num_tags, end_tag].data +
                    -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self.constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_len + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            # perform viterbi decoding sample by sample
            seq_len = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(seq_len + 1), :num_tags] = prediction[:seq_len]
            # And at the last timestep we must have the END_TAG
            tag_sequence[seq_len + 1, end_tag] = 0.
            viterbi_path = viterbi_decode(tag_sequence[:(seq_len + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append(viterbi_path)
        return best_paths


class XLNetABSATagger():
    pass
