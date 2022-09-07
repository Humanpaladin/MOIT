from fastNLP.io import Pipe, DataBundle, Loader
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer, BertTokenizer
import numpy as np
from itertools import chain


def cmp_aspect(v1, v2):
    if v1[0]['from'] == v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']


def cmp_opinion(v1, v2):
    if v1[1]['from'] == v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class BartBPEABSAPipe(Pipe):
    def __init__(self, tokenizer_path=None, opinion_first=False):
        super(BartBPEABSAPipe, self).__init__()
        assert opinion_first is False
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 2. 构建一个词典，真正写入到 tokenizer 的 vocab 中的是词典的 value
        self.mapping = {
            'tag': '<<tag>>',       # 注意: 真正加入 vocab 的 token 是: <<tag>>
        }

        self.opinion_first = opinion_first
        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        # 3. 构建待加入 tokenizer 词表中的 token 的列表
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x: len(x), reverse=True)
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x: len(x), reverse=True)

        # 4. 判断待加入的 token 是否是初始词表中的 unk token
        for token in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([token])[0] == self.tokenizer.unk_token_id

        # 5. 将待加入的 token (<<tag>>) 加入到 tokenizer 的 vocab 中，以及将 <<tag>> 作为一个不可分割的 token
        sorted_add_tokens.extend(["<s>", "</s>", "<unk>", "<pad>", "<mask>"])
        self.tokenizer.add_tokens(sorted_add_tokens)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        self.tokenizer.unique_no_split_tokens = list(set(unique_no_split_tokens + sorted_add_tokens))

        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.mask_token = "<mask>"

        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, "{}'s length is not 1".format(value)
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        target_shift = len(self.mapping) + 2
        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            tokens_list = []
            for word in raw_words:
                tokens = self.tokenizer.tokenize(word)
                tokens_list.append(tokens)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                word_bpes.append(token_ids)
            word_bpes.append([self.tokenizer.eos_token_id])
            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(lens).tolist()


            # todo 这里需要解决一个当 opinion first 的时候，按照 opinion 排序，否则按照 aspects 排序
            aspects = ins['aspects']

            assert len(aspects) == 1
            aspects = aspects[0]
            opinions = ins['opinions']
            _word_bpes = list(chain(*word_bpes))
            a_bpes = None
            o_bpes = []
            target_spans = []
            target = [0]
            a_start_bpe = cum_lens[aspects['from']]
            a_end_bpe = cum_lens[aspects['to'] - 1]
            a_bpes = [a_start_bpe+target_shift, a_end_bpe+target_shift]
            target.extend(a_bpes)
            for idx, word in zip((a_start_bpe, a_end_bpe), (aspects['term'][0], aspects['term'][-1])):
                print('a_start_bpe:', a_start_bpe)
                print('a_end_bpe:', a_end_bpe)
                print("aspects['term'][0]:", aspects['term'][0])
                print("aspects['term'][-1]:", aspects['term'][-1])
                assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word)[:1])[0] or _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word)[-1:])[0]

            for opinion in opinions:
                o_start_bpe = cum_lens[opinion['from']]
                o_end_bpe = cum_lens[opinion['to']-1]
                o_bpes.extend([o_start_bpe+target_shift, o_end_bpe+target_shift, target_shift-1])
                # 这里需要evaluate是否是对齐的
                for idx, word in zip((o_start_bpe, o_end_bpe), (opinion['term'][0], opinion['term'][-1])):
                    assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word)[:1])[0] or _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word)[-1:])[0]
                target_spans.append((o_start_bpe+target_shift, o_end_bpe+target_shift))             # [(11, 11)]   ->   [(11, 11), (15, 15)]
            target.extend(o_bpes)                                                                   # [0, 5, 7, 11, 11, 2, 15, 15, 2]
            target.append(1)  # append 1是由于特殊的eos                                               # [0, 5, 7, 11, 11, 2, 15, 15, 2, 1]
            result = {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes))}     # target:                   [0, 5, 7, 11, 11, 2, 15, 15, 2, 1]
                                                                                                                    # target_spans:             [(11, 11), (15, 15)]
                                                                                                                    # list(chain(*word_bpes)):  [0, 20, 2131, 16708, 5884, 8, 19464, 58, 19066, 8, 14166, 7, 19858, 2156, 8, 5, 17988, 6255, 5488, 16, 372, 13, 1144, 906, 24959, 5110, 479, 2]

            return result

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Preparing target.')
        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'tgt_seq_len', 'src_tokens', 'src_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        fan_absa_loader = FanABSALoader(demo=demo)
        data_bundle = fan_absa_loader.load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle


class FanABSALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        print("path:", path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # TODO 这里首先需要把数据merge到一起，相同的input的
        new_data = {}
        merge_line = 0
        if 'train' in path:
            for d in data:
                assert len(d['aspects']) == 1
                key = (d['raw_words'], d['aspects'][0]['from'], d['aspects'][0]['to'])
                if key in new_data:
                    print(new_data[key])
                    merge_line += 1
                    new_data[key]['opinions'].extend(d['opinions'])
                else:
                    new_data[key] = d
            new_data = list(new_data.values())
        else:
            new_data = data

        ds = DataSet()
        for ins in new_data:
            assert isinstance(ins, dict)
            assert len(ins) == 5
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            ins = Instance(raw_words=tokens, aspects=aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds) > 30:
                break
        print(f"Merge {merge_line} lines from old:{len(data)} to new:{len(new_data)}.")
        return ds



