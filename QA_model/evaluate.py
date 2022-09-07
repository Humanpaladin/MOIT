""" Customized version of the official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

import prettytable as pt
import numpy as np


def BiDAF_compute_metrics(dataset, all_b_e_s_pred_triplets):
    """
    要将 dataset, predictions 转换为 all_pred_triplets, all_gold_triplets
    """
    aspect_ch_en = {
        "内饰": "NS",
        "动力": "DL",
        "外观": "WG",
        "性价比": "XJB",
        "操控": "CK",
        "空间": "KJ",
        "能耗": "NH",
        "舒适性": "SSX"
    }

    all_b_e_s_gold_triplets = []

    with open(".data/data_IOE/normal/dev-v0.1.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            example = json.loads(line)
            aspect = aspect_ch_en[example["question"]]
            start_idx = example["s_idx"]
            end_idx = example["e_idx"]
            triplet = [(start_idx, end_idx, aspect)]
            all_b_e_s_gold_triplets.append(triplet)

    total_aspect_triplets_distribution_gold = np.zeros(8)
    total_aspect_triplets_distribution_pred = np.zeros(8)
    total_aspect_triplets_distribution_hit = np.zeros(8)


    aspect_triplets_distribution_recall = np.zeros(8)
    aspect_triplets_distribution_precision = np.zeros(8)
    aspect_triplets_distribution_f1 = np.zeros(8)

    num_examples = len(all_b_e_s_pred_triplets)
    for i in range(num_examples):
        pred_b_e_a_triplets = all_b_e_s_pred_triplets[i]
        gold_b_e_a_triplets = all_b_e_s_gold_triplets[i]

        aspect_triplets_distribution_hit, aspect_triplets_distribution_gold, aspect_triplets_distribution_pred = count_aspect_triplets_distribution(
            gold_b_e_a_triplets, pred_b_e_a_triplets)


        total_aspect_triplets_distribution_gold += aspect_triplets_distribution_gold
        total_aspect_triplets_distribution_pred += aspect_triplets_distribution_pred
        total_aspect_triplets_distribution_hit += aspect_triplets_distribution_hit
    for i in range(8):
        gold = total_aspect_triplets_distribution_gold[i]
        pred = total_aspect_triplets_distribution_pred[i]
        hit = total_aspect_triplets_distribution_hit[i]

        aspect_triplets_distribution_precision[i] = round(float(hit) / float(pred + 0.0000001), 4)
        aspect_triplets_distribution_recall[i] = round(float(hit) / float(gold + 0.0000001), 4)
        aspect_triplets_distribution_f1[i] = round(2 * aspect_triplets_distribution_precision[i] * aspect_triplets_distribution_recall[i] / (aspect_triplets_distribution_precision[i] + aspect_triplets_distribution_recall[i] + 0.0000001), 4)

    # 打印各个 aspect 对应的 precision, recall, f1
    precision_row = ['precision']
    recall_row = ['recall']
    f1_row = ['f1']

    precision_row.extend(aspect_triplets_distribution_precision.tolist())
    recall_row.extend(aspect_triplets_distribution_recall.tolist())
    f1_row.extend(aspect_triplets_distribution_f1.tolist())

    table = pt.PrettyTable()
    table.title = 'Performance by aspects'
    table.field_names = ['', "NS", "DL", "WG", "XJB", "CK", "KJ", "NH", "SSX"]
    table.add_row(f1_row)
    print("\n", table)

    macro_f1 = round(aspect_triplets_distribution_f1.mean(), 4)

    # 计算 micro f1
    num_gold_total = sum(total_aspect_triplets_distribution_gold)
    num_pred_total = sum(total_aspect_triplets_distribution_pred)
    num_tp_total = sum(total_aspect_triplets_distribution_hit)
    micro_precision = round(float(num_tp_total) / float(num_pred_total + 0.00000001), 4)
    micro_recall = round(float(num_tp_total) / float(num_gold_total + 0.00000001), 4)
    micro_f1 = round(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 0.00000001), 4)

    scores = {
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }

    return scores


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if str(qa['id']) not in predictions:
                    message = 'Unanswered question ' + str(qa['id']) + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[str(qa['id'])]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def main(args, all_b_e_s_pred_triplets):
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    results = BiDAF_compute_metrics(dataset, all_b_e_s_pred_triplets)
    return results


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
