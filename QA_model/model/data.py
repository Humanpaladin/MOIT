# -*- coding: UTF-8 -*-

import json
import os
import nltk
import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

from torchtext import data
# from torchtext import datasets
from torchtext.vocab import GloVe


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id
unk_token_id = tokenizer.unk_token_id

max_input_length = 512


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[0: max_input_length-2]
    return tokens


class SQuAD():
    def __init__(self, args):
        data_dir = '.data/data_IOE/normal'
        dataset_dir = data_dir + '/torchtext/'
        train_examples = dataset_dir + 'train_examples.pt'
        dev_examples = dataset_dir + 'dev_examples.pt'

        print("converting json to jsonl files ...")
        if not os.path.exists('{}/{}l'.format(data_dir, args.train_file)):                              # 如果不存在 ./data/data_IOE/normal/train-v0.1.jsonl，则创建
            print('{}/{}l does not exist, creating ...'.format(data_dir, args.train_file))
            self.preprocess_file('{}/{}'.format(data_dir, args.train_file))
        if not os.path.exists('{}/{}l'.format(data_dir, args.dev_file)):                                # 如果不存在 ./data/data_IOE/normal/dev-v0.1.jsonl，则创建
            print('{}/{}l does not exist, creating ...'.format(data_dir, args.dev_file))
            self.preprocess_file('{}/{}'.format(data_dir, args.dev_file))

        self.RAW = data.RawField()
        self.RAW.is_target = False

        # 用于处理字符级别
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=tokenize_and_cut)

        # self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.WORD = data.Field(
            batch_first=True,
            use_vocab=False,
            tokenize=tokenize_and_cut,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=cls_token_id,
            eos_token=sep_token_id,
            pad_token=pad_token_id,
            unk_token=unk_token_id,
            lower=True,
            include_lengths=True,
            # fix_length=214
        )
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {
            'id': ('id', self.RAW),
            's_idx': ('s_idx', self.LABEL),
            'e_idx': ('e_idx', self.LABEL),
            'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
            'question': [('q_word', self.WORD), ('q_char', self.CHAR)]
        }

        list_fields = [
            ('id', self.RAW),
            ('s_idx', self.LABEL),
            ('e_idx', self.LABEL),
            ('c_word', self.WORD),
            ('c_char', self.CHAR),
            ('q_word', self.WORD),
            ('q_char', self.CHAR)
        ]

        if os.path.exists(train_examples) and os.path.exists(dev_examples):    # 由 jsonl 文件得到的数据集文件，存放在 '.data/squad/torchtext/' 中

            print("loading train-dev splits...")
            train_examples_list = torch.load(train_examples)
            dev_examples_list = torch.load(dev_examples)

            self.train = data.Dataset(examples=train_examples_list, fields=list_fields)     # 将得到 Example() 类对象的列表转换为 Dataset() 类对象
            self.dev = data.Dataset(examples=dev_examples_list, fields=list_fields)         # 将得到 Example() 类对象的列表转换为 Dataset() 类对象

        else:
            print("building train-dev datasets...")

            # 1. 构建 TabularDataset() 类型的数据集对象
            self.train, self.dev = data.TabularDataset.splits(
                path=data_dir,
                train='{}l'.format(args.train_file),
                validation='{}l'.format(args.dev_file),
                format='json',
                fields=dict_fields
            )

            # 2. 将 TabularDataset() 类对象转换为 torchtext.data.dataset.Dataset()类对象，便于后面代码都统一使用 Dataset() 类数据类型
            self.train = data.Dataset(examples=self.train.examples, fields=list_fields)
            self.dev = data.Dataset(examples=self.dev.examples, fields=list_fields)

            # 3. 取得 torchtext.data.dataset.Dataset() 类对象的 examples 属性 (为 list of Example() 类对象)，其为序列化数据，可以使用 torch.save() 存储，供下次需要
            #    使用数据集时直接调用
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            print("saving train-dev datasets...")
            train_examples_list = self.train.examples
            dev_examples_list = self.dev.examples
            torch.save(train_examples_list, train_examples)
            torch.save(dev_examples_list, dev_examples)

        # cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building char vocabs...")
        self.CHAR.build_vocab(self.train, self.dev)
        print('self.CHAR.vocab.itos:', self.CHAR.vocab.itos)

        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        # self.train_iter = data.BucketIterator(
        self.train_iter = data.Iterator(
            self.train,
            batch_size=args.train_batch_size,
            device=device,
            repeat=True,
            shuffle=True,
            sort_key=lambda x: len(x.c_word)
        )

        # self.dev_iter = data.BucketIterator(
        self.dev_iter = data.Iterator(
            self.dev,
            batch_size=args.dev_batch_size,
            device=device,
            repeat=False,
            shuffle=False,
            sort_key=lambda x: len(x.c_word)
        )


    def preprocess_file(self, file_name):           # .json 文件 --> .jsonl 文件
        max_seq_length = -1
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']          # 每一个段落的内容
                    tokens = tokenize_and_cut(context)

                    if len(tokens) > max_seq_length:
                        max_seq_length = len(tokens)

                    for qa in paragraph['qas']:             # 针对段落的每一个问题及其答案
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:           # train 和 dev 中 answers 列表中的元素均可为多个
                            answer = ans['text']
                            s_idx = ans['answer_start'] + 1     # 这里因为在每个句子前会加一个 [SEP]，所以将 s_idx, e_idx 分别后移一位
                            e_idx = s_idx + len(tokenizer.tokenize(answer))
                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))

        with open('{}l'.format(file_name), 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f, ensure_ascii=False)
        print("{} created with {} records in.".format(file_name+'l', len(dump)))
        print("max_seq_len is {}".format(max_seq_length))
