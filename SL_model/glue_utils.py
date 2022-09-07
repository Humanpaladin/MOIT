from __future__ import absolute_import, division, print_function
import csv
import logging
import math
import os
import sys
import time
from io import open
import numpy as np
import torch
from matplotlib import pyplot as plt
from seq_utils import *
from prettytable import PrettyTable
logger = logging.getLogger(__name__)


class SingleRawExample(object):     # 原来的 InputExample() 类
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class SequentializedSingleRawExample(object):   # 原来的 SeqInputFeatures() 类
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.evaluate_label_ids = evaluate_label_ids


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        """
        为训练集取得一些 SingleRawExample 对象
        Gets a collection of 'SingleRawExample's for the train set.
        """
        print('data_dir:', data_dir)
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        为 dev 集取得一些 SingleRawExample 对象
        Gets a collection of 'SingleRawExample's for the dev set.
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        为 test 集取得一些 SingleRawExample 类对象
        Gets a collection of 'SingleRawExample's for the test dev.
        """
        raise NotImplementedError()

    def get_all_possible_labels(self):
        """
        Gets the list of labels for the data set.
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class DataSetToSingleRawExampleList(DataProcessor):
    def get_train_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='train', tagging_schema=tagging_schema)

    def get_dev_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='dev', tagging_schema=tagging_schema)

    def get_test_examples(self, data_dir, tagging_schema):
        return self._create_examples(data_dir=data_dir, set_type='test', tagging_schema=tagging_schema)

    def get_all_possible_labels(self, tagging_schema):
        if tagging_schema == 'OT':
            return []
        elif tagging_schema == 'BIO':
            return [
                'O', 'EQ',
                'B-NS', 'I-NS',
                'B-DL', 'I-DL',
                'B-WG', 'I-WG',
                'B-XJB', 'I-XJB',
                'B-CK', 'I-CK',
                'B-KJ', 'I-KJ',
                'B-NH', 'I-NH',
                'B-SSX', 'I-SSX'
            ]
        elif tagging_schema == 'BIEOS':
            real_labels = [
                'O', 'EQ',
                'B-NS', 'I-NS', 'E-NS', 'S-NS',
                'B-DL', 'I-DL', 'E-DL', 'S-DL',
                'B-WG', 'I-WG', 'E-WG', 'S-WG',
                'B-XJB', 'I-XJB', 'E-XJB', 'S-XJB',
                'B-CK', 'I-CK', 'E-CK', 'S-CK',
                'B-KJ', 'I-KJ', 'E-KJ', 'S-KJ',
                'B-NH', 'I-NH', 'E-NH', 'S-NH',
                'B-SSX', 'I-SSX', 'E-SSX', 'S-SSX'
            ]
            logger.info('The number of all labels is %d' % len(real_labels))
            logger.info('All the labels are %s', real_labels)
            return real_labels
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)


def convert_SingleRawExamples_to_SequentializedSingleRawExamples(
        single_raw_examples,
        all_possible_labels,
        tokenizer,
        cls_token_at_end=False,
        pad_on_left=False,
        cls_token='[CLS]',
        sep_token='[SEP]',
        pad_token=0,
        sequence_a_segment_id=0,
        sequence_b_segment_id=1,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
):

    label_map = {label: i for i, label in enumerate(all_possible_labels)}
    sequentialized_raw_single_examples = []     # 要返回的 SequentializedSingleRawExample 类对象的列表

    tokenized_single_raw_examples = []  # 为 SingleRawExample 对象转换为 SequentializedSingleRawExample 对象的一个中间变量

    max_seq_length = 231
    for (ex_index, single_raw_example) in enumerate(single_raw_examples):  # 对于每一个样本句子
        tokens_a = []
        labels_a = []
        evaluate_label_ids = []
        words = single_raw_example.text_a.split(' ')
        word_id_before_tokenization = 0
        word_id_after_tokenization = 0

        # 下面这个 for 循环整体上处理一个句子级别。将句子中的单词和 label 打包起来一起处理
        for word, label in zip(words, single_raw_example.labels):
            # 处理每一个单词得到的 token、token 对应的 label、记录有效 token 的位置
            sub_words = tokenizer.tokenize(word)
            tokens_a.extend(sub_words)
            if label != 'O':
                labels_a.extend([label] + ['EQ'] * (len(sub_words) - 1))
            else:
                labels_a.extend(['O'] * len(sub_words))
            evaluate_label_ids.append(word_id_after_tokenization)

            word_id_before_tokenization += 1
            word_id_after_tokenization += len(sub_words)

        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        assert word_id_after_tokenization == len(tokens_a)

        tokenized_single_raw_example = (tokens_a, labels_a, evaluate_label_ids)
        tokenized_single_raw_examples.append(tokenized_single_raw_example)

        if len(tokens_a) > max_seq_length:
            max_seq_length = len(tokens_a)

    # 第二个 for 循环，主要是将中间变量 tokenized_single_example 转换为 SequentializedSingleRawExample
    logger.info('maximal sequence length is %d' % max_seq_length)
    for ex_index, tokenized_single_raw_example in enumerate(tokenized_single_raw_examples):     # 对于每一个三元组，即每一个样本句子
        tokens_a = tokenized_single_raw_example[0]
        labels_a = tokenized_single_raw_example[1]
        evaluate_label_ids = tokenized_single_raw_example[2]
        tokens = tokens_a + [sep_token]
        labels = labels_a + ['O']
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            # evaluate label ids not change
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            labels = labels + ['O']
        else:
            # right shift 1 for evaluate_label_ids
            tokens = [cls_token] + tokens
            labels = ['O'] + labels
            segment_ids = [cls_token_segment_id] + segment_ids
            evaluate_label_ids += 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_map[label] for label in labels]
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            label_ids = ([0] * padding_length) + label_ids

            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            evaluate_label_ids += padding_length
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            label_ids = label_ids + ([0] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        if len(input_ids) > 231:
            print(tokens_a)
        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("example (an SingleRawExample object): %s" % tokenized_single_raw_example[0])
            logger.info("ex_index: %s" % ex_index)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s " % ' '.join([str(x) for x in label_ids]))
            logger.info("evaluate label ids: %s" % evaluate_label_ids)

        sequentialized_raw_single_examples.append(
            SequentializedSingleRawExample(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                evaluate_label_ids=evaluate_label_ids
            )
        )
    return sequentialized_raw_single_examples


def count_aspect_triplets_distribution(gold_b_e_a_triplets, pred_b_e_a_triplets):
    aspect_map = {
        'NS': 0,
        'DL': 1,
        'WG': 2,
        'XJB': 3,
        'CK': 4,
        'KJ': 5,
        'NH': 6,
        'SSX': 7
    }
    num_aspects = len(list(aspect_map.keys()))
    gold_count = np.zeros(num_aspects)
    pred_count = np.zeros(num_aspects)
    hit_count = np.zeros(num_aspects)

    for triplet in gold_b_e_a_triplets:
        aspect = triplet[2]
        aspect_id = aspect_map[aspect]
        gold_count[aspect_id] += 1
    for triplet in pred_b_e_a_triplets:
        aspect = triplet[2]
        aspect_id = aspect_map[aspect]
        pred_count[aspect_id] += 1
        if triplet in gold_b_e_a_triplets:
            hit_count[aspect_id] += 1

    return hit_count, gold_count, pred_count


def compute_metrics(all_pred_label_ids, all_gold_label_ids, all_evaluate_label_ids, tagging_schema):
    # 1. 构建 sentiment_label_map 字典，即每个 token 所可能对应的 label 的字典。
    if tagging_schema == 'BIEOS':
        aspect_label_map = {
            'O': 0, 'EQ': 1,
            'B-NS': 2, 'I-NS': 3, 'E-NS': 4, 'S-NS': 5,
            'B-DL': 6, 'I-DL': 7, 'E-DL': 8, 'S-DL': 9,
            'B-WG': 10, 'I-WG': 11, 'E-WG': 12, 'S-WG': 13,
            'B-XJB': 14, 'I-XJB': 15, 'E-XJB': 16, 'S-XJB': 17,
            'B-CK': 18, 'I-CK': 19, 'E-CK': 20, 'S-CK': 21,
            'B-KJ': 22, 'I-KJ': 23, 'E-KJ': 24, 'S-KJ': 25,
            'B-NH': 26, 'I-NH': 27, 'E-NH': 28, 'S-NH': 29,
            'B-SSX': 30, 'I-SSX': 31, 'E-SSX': 32, 'S-SSX': 33}
    elif tagging_schema == 'BIO':
        raise NotImplementedError()
    elif tagging_schema == 'OT':
        raise NotImplementedError()
    else:
        raise Exception("Invalid tagging schema %s..." % tagging_schema)

    aspect_label_id_2_aspect_label = {}
    for aspect_label in aspect_label_map:
        aspect_label_id = aspect_label_map[aspect_label]
        aspect_label_id_2_aspect_label[aspect_label_id] = aspect_label

    total_aspect_triplets_distribution_gold = np.zeros(8)
    total_aspect_triplets_distribution_pred = np.zeros(8)
    total_aspect_triplets_distribution_hit = np.zeros(8)

    aspect_triplets_distribution_recall = np.zeros(8)
    aspect_triplets_distribution_precision = np.zeros(8)
    aspect_triplets_distribution_f1 = np.zeros(8)

    num_examples = len(all_evaluate_label_ids)
    for i in range(num_examples):   # 对于每一个样本，处理其 pred label id 和 gold label id
        evaluate_label_ids = all_evaluate_label_ids[i]
        valid_pred_label_ids = all_pred_label_ids[i][evaluate_label_ids]
        valid_gold_label_ids = all_gold_label_ids[i][evaluate_label_ids]
        assert len(valid_pred_label_ids) == len(valid_gold_label_ids)

        valid_pred_labels = [aspect_label_id_2_aspect_label[label_id] for label_id in valid_pred_label_ids]  # 得到一个句子的有效 token 对应的 pred label，可计算其包含的 pred triplets
        valid_gold_labels = [aspect_label_id_2_aspect_label[label_id] for label_id in valid_gold_label_ids]  # 得到一个句子的有效 token 对应的 gold label，可计算其包含的 gold triplets

        pred_b_e_a_triplets = labels_2_b_e_a_triplets(valid_pred_labels)
        gold_b_e_a_triplets = labels_2_b_e_a_triplets(valid_gold_labels)

        aspect_triplets_distribution_hit, aspect_triplets_distribution_gold, aspect_triplets_distribution_pred = count_aspect_triplets_distribution(gold_b_e_a_triplets, pred_b_e_a_triplets)

        total_aspect_triplets_distribution_gold += aspect_triplets_distribution_gold  # 统计所有句子包含的 triplets 的分布 (gold)
        total_aspect_triplets_distribution_pred += aspect_triplets_distribution_pred  # 统计所有句子包含的 triplets 的分布 (pred)
        total_aspect_triplets_distribution_hit += aspect_triplets_distribution_hit    # 统计所有句子包含的 triplets 的分布 (hit)

    # 以下开始计算指标
    for i in range(8):     # 分别考虑每种 aspect 类型的指标
        gold = total_aspect_triplets_distribution_gold[i]    # 某一个位置的情感类型 (某一种情感类型的 triplets)，其 gold 三元组有多少个
        pred = total_aspect_triplets_distribution_pred[i]    # 某一个位置的情感类型 (某一种情感类型的 triplets)，其 pred 三元组有多少个
        hit = total_aspect_triplets_distribution_hit[i]      # 某一个位置的情感类型 (某一种情感类型的 triplets)，某命中了多少个

        aspect_triplets_distribution_precision[i] = round(float(hit) / float(pred + 0.0000001), 4)    # 某一个位置的情感类型 (某一种情感类型的 triplets)，其预测的 precision
        aspect_triplets_distribution_recall[i] = round(float(hit) / float(gold + 0.0000001), 4)       # 某一个位置的情感类型 (某一种情感类型的 triplets)，其预测的 recall
        aspect_triplets_distribution_f1[i] = round(2 * aspect_triplets_distribution_precision[i] * aspect_triplets_distribution_recall[i] / (aspect_triplets_distribution_precision[i] + aspect_triplets_distribution_recall[i] + 0.0000001), 4)

    # 打印各个 aspect 对应的 precision, recall, f1
    precision_row = ['precision']
    recall_row = ['recall']
    f1_row = ['f1']

    precision_row.extend(aspect_triplets_distribution_precision.tolist())
    recall_row.extend(aspect_triplets_distribution_recall.tolist())
    f1_row.extend(aspect_triplets_distribution_f1.tolist())

    table = PrettyTable()
    table.title = 'Performance by aspects'
    table.field_names = ['', "NS", "DL", "WG", "XJB", "CK", "KJ", "NH", "SSX"]
    table.add_row(precision_row)
    table.add_row(recall_row)
    table.add_row(f1_row)
    print(table)


    macro_f1 = round(aspect_triplets_distribution_f1.mean(), 4)

    # 计算 micro f1
    num_gold_total = sum(total_aspect_triplets_distribution_gold)
    num_pred_total = sum(total_aspect_triplets_distribution_pred)
    num_tp_total = sum(total_aspect_triplets_distribution_hit)

    micro_precision = round(float(num_tp_total) / float(num_pred_total + 0.00000001), 4)
    micro_recall = round(float(num_tp_total) / float(num_gold_total + 0.00000001), 4)
    micro_f1 = round(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 0.00000001), 4)

    scores = {
        'macro-f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro-f1': micro_f1
    }

    return scores


dataset_to_single_raw_example_list_processors = {
    "laptop14": DataSetToSingleRawExampleList,
    "rest_total": DataSetToSingleRawExampleList,
    "auto_home": DataSetToSingleRawExampleList,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "rest14": "classification",
    "rest15": "classification",
    "rest16": "classification",
    "laptop14": "classification",
    "rest_total": "classification",
    "auto_home": "classification",
}


def get_boundary_all_possible_labels():
    boundary_all_possible_labels = ['O', 'EQ', 'B', 'I', 'E', 'S']
    boundary_label_map = {boundary_label: i for i, boundary_label in enumerate(boundary_all_possible_labels)}   # {'O': 0, 'EQ': 1, 'B': 2, 'I': 3, 'E': 4, 'S': 5}
    boundary_label_id_to_label = {v: k for k, v in boundary_label_map.items()}                                  # {0: 'O', 1: 'EQ', 2: 'B', 3: 'I', 4: 'E', 5: 'S'}
    return tuple([boundary_all_possible_labels, boundary_label_map, boundary_label_id_to_label])


def get_aspect_all_possible_labels():
    aspect_all_possible_labels = ['O', 'EQ', 'NS', 'DL', 'WG', 'XJB', 'CK', 'KJ', 'NH', 'SSX']
    aspect_label_map = {aspect_label: i for i, aspect_label in enumerate(aspect_all_possible_labels)}
    aspect_label_id_to_label = {v: k for k, v in aspect_label_map.items()}  # {0: 'O', 1: 'EQ', 2: 'DL', 3: 'KJ', 4: 'NS', 5: 'WG', 6: 'CK', 7: 'SSX', 8: 'NH', 9: 'XJB'}
    return tuple([aspect_all_possible_labels, aspect_label_map, aspect_label_id_to_label])


def get_polarity_all_possible_labels():
    polarity_all_possible_labels = ['O', 'EQ', 'POS', 'NEG']
    polarity_label_map = {polarity_label: i for i, polarity_label in enumerate(polarity_all_possible_labels)}
    polarity_label_id_to_label = {v: k for k, v in polarity_label_map.items()}  # {0: 'O', 1: 'EQ', 2: 'POS', 3: 'NEG'}
    return tuple([polarity_all_possible_labels, polarity_label_map, polarity_label_id_to_label])
