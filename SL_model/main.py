# -*- coding: UTF-8 -*-
import argparse
import datetime
import glob
import logging
import os
import random
from prettytable import PrettyTable

import numpy as np
from tensorboardX.writer import SummaryWriter
from torch.utils.data import SequentialSampler
from tqdm import trange, tqdm

#
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import RandomSampler

# 以下为 transformer 中从具体的文件 (可循迹的文件) 中导入相应的包
import torch
from transformers.configuration_bert import BertConfig
from transformers.configuration_xlnet import XLNetConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.file_utils import WEIGHTS_NAME
from transformers.optimization import AdamW, WarmupLinearSchedule

from absa_layer import BertABSATagger, XLNetABSATagger

from glue_utils import dataset_to_single_raw_example_list_processors, compute_metrics
from glue_utils import convert_SingleRawExamples_to_SequentializedSingleRawExamples

np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetABSATagger, XLNetTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    # --model_type 为模型的类型: bert 或者 xlnet
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected from the list:" + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--absa_type", default=None, type=str, required=True,
                        help="Downstream absa layer type selected from the list: [linear, gru, san, tfm, crf]")

    # 是否 finetune
    parser.add_argument("--tfm_mode", default=None, type=str, required=True,
                        help="mode of the pre-trained transformer, selected from: [finetune]")

    # 是否固定 transformers 的参数
    parser.add_argument("--fix_tfm", default=None, type=int, required=True,
                        help="whether fix the transformer parameters or not")

    # 预训练模型的名字，如: bert-base-uncased, bert-base-chinese 等
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected from the list:" + ", ".join(
                            ALL_MODELS))

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected from the list: " + ", ".join(
                            dataset_to_single_raw_example_list_processors.keys()))

    # other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length", default=231, type=int,
                        help="The maximum total input sequence length after Tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. "
                             "Override num_train_epochs.")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")

    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tagging_schema', type=str, default='BIEOS')

    parser.add_argument("--overfit", type=int, default=0, help="if evaluate overfit or not")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)

    args = parser.parse_args()

    output_dir = "%s-%s-%s-%s" % (args.model_type, args.absa_type, args.task_name, args.tfm_mode)
    if args.fix_tfm:
        output_dir = "%s-fix" % output_dir
    if args.overfit:
        output_dir = "%s-overfit" % output_dir
        args.max_steps = 6000
    args.output_dir = output_dir

    return args


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    set_seed(args)

    train_dataset_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_dataset_sampler,
        batch_size=args.train_batch_size
    )

    if args.max_steps:
        max_batch_load_times = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        logger.info("args.num_train_epochs: %d" % args.num_train_epochs)
    else:
        max_batch_load_times = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']

    grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_batch_load_times)

    logger.info("=== Running training ===")
    logger.info("   Number of examples = %d", len(train_dataset))
    logger.info("   Number of examples = %d", len(train_dataset.tensors))
    logger.info("   Number of epochs = %d", args.num_train_epochs)
    logger.info("   Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("   Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("   Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("   Total optimization steps (Total batch load times) = %d\n", max_batch_load_times)

    global_batch_load_counter = 0
    global_batch_loss_accumulation = 0.0
    logging_loss = 0.0
    model.zero_grad()
    epoch_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
        position=0
    )

    for epoch_counter in epoch_iterator:
        batch_iterator = tqdm(
            train_dataloader,
            desc="\tBatch (Iteration or Step)",
            disable=args.local_rank not in [-1, 0],
            position=0,
        )

        for batch_counter, batch in enumerate(batch_iterator):
            batch = tuple(tensor.to(args.device) for tensor in batch)
            batch_inputs = {
                "input_ids": batch[0],
                "input_masks": batch[1],
                "segment_ids": batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                "label_ids": batch[3],
            }
            batch_outputs = model(**batch_inputs)
            batch_loss = None

            if args.n_gpu > 1:
                batch_loss = batch_outputs[0].mean()

            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            global_batch_loss_accumulation += batch_loss.item()
            batch_average_loss = global_batch_loss_accumulation / (global_batch_load_counter + 1)

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (global_batch_load_counter + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_batch_load_counter += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_batch_load_counter % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        print('\tduring_training results:', results)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_batch_load_counter)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_batch_load_counter)
                    tb_writer.add_scalar('loss', (global_batch_loss_accumulation - logging_loss) / args.logging_steps, global_batch_load_counter)
                    logging_loss = global_batch_loss_accumulation

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_batch_load_counter % args.save_steps == 0:
                    checkpoint_saving_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_batch_load_counter))
                    if not os.path.exists(checkpoint_saving_dir):
                        os.makedirs(checkpoint_saving_dir)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(checkpoint_saving_dir)
                    torch.save(args, os.path.join(checkpoint_saving_dir, 'training_args.bin'))

            if args.max_steps > 0 and global_batch_load_counter > args.max_steps:
                batch_iterator.close()
                break

        if args.max_steps > 0 and global_batch_load_counter > args.max_steps:
            epoch_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_batch_load_counter, global_batch_loss_accumulation / global_batch_load_counter


def evaluate(args, model, tokenizer, mode, prefix=""):
    results = {}
    eval_dataset, dataset_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode=mode)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    logger.info("===== Running evaluation on %s.txt =====" % mode)
    eval_dataset_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_dataset_sampler, batch_size=args.eval_batch_size)
    batch_loss_accumulation = 0.0
    batch_load_counter = 0
    dataset_logits_numpy = None
    dataset_gold_label_ids_numpy = eval_dataset.tensors[3].numpy()
    crf_logits, crf_mask = [], []
    model = torch.nn.parallel.DataParallel(model)

    for batch in tqdm(eval_dataloader, desc="\tEvaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            batch_inputs = {
                'input_ids': batch[0],
                'input_masks': batch[1],
                'segment_ids': batch[2],
                'label_ids': batch[3],
            }

            batch_outputs = model(**batch_inputs)
            batch_loss_accumulation = batch_outputs[0].mean().item()
            batch_logits = batch_outputs[1]
        batch_load_counter += 1

        if dataset_logits_numpy is None:
            dataset_logits_numpy = batch_logits.detach().cpu().numpy()  # (9286, 102, 10)
        else:
            dataset_logits_numpy = np.append(dataset_logits_numpy, batch_logits.detach().cpu().numpy(), axis=0)  # np.ndarray(9286, 102, 4)

    batch_average_loss = batch_loss_accumulation / batch_load_counter

    if model.module.tagger_config.absa_type != 'crf':
        dataset_pred_label_ids_numpy = np.argmax(dataset_logits_numpy, axis=-1)
    else:
        crf_logits = torch.cat(crf_logits, dim=0)
        crf_mask = torch.cat(crf_mask, dim=0)
        dataset_pred_label_ids_numpy = model.tagger.viterbi_tags(logits=crf_logits, mask=crf_mask)

    result = compute_metrics(
        dataset_pred_label_ids_numpy,
        dataset_gold_label_ids_numpy,
        dataset_evaluate_label_ids,
        args.tagging_schema
    )
    result['batch_average_loss'] = round(batch_average_loss, 6)
    results.update(result)

    row_metrics = list(results.keys())
    row_results = list(results.values())

    table = PrettyTable()
    table.title = mode + ' Performance over' + ' checkpoint-'+prefix
    table.field_names = row_metrics
    table.add_row(row_results)
    print(table)

    output_eval_file = os.path.join(args.output_dir, "%s_results.txt" % mode)

    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            if 'batch_average_loss' in key:
                pass
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenzier, mode="train"):
    """
    功能:
        完成将原始数据集向模型处理的数据的转换:
            dataset --> single_raw_examples --> sequentialized_single_raw_examples --> tensors --> tensor_dataset
    """
    dataset_to_single_raw_example_list_processor = dataset_to_single_raw_example_list_processors[task]()
    cached_sequentialized_single_raw_examples_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)
        )
    )

    if os.path.exists(cached_sequentialized_single_raw_examples_file):
        sequentialized_single_raw_examples = torch.load(cached_sequentialized_single_raw_examples_file)

    else:
        logger.info("Creating sequentialized_raw_single_examples from dataset file in %s", args.data_dir)

        all_possible_labels = dataset_to_single_raw_example_list_processor.get_all_possible_labels(tagging_schema=args.tagging_schema)

        if mode == "train":
            single_raw_examples = dataset_to_single_raw_example_list_processor.get_train_examples(args.data_dir, tagging_schema=args.tagging_schema)
        elif mode == "dev":
            single_raw_examples = dataset_to_single_raw_example_list_processor.get_dev_examples(args.data_dir, tagging_schema=args.tagging_schema)
        elif mode == "test":
            single_raw_examples = dataset_to_single_raw_example_list_processor.get_test_examples(args.data_dir, tagging_schema=args.tagging_schema)
        else:
            raise Exception("Invalid mode ".format(mode))

        sequentialized_single_raw_examples = convert_SingleRawExamples_to_SequentializedSingleRawExamples(
            single_raw_examples=single_raw_examples,
            all_possible_labels=all_possible_labels,
            tokenizer=tokenzier,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            pad_on_left=bool(args.model_type in ["xlnet"]),
            cls_token=tokenzier.cls_token,
            sep_token=tokenzier.sep_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving sequentialized_single_raw_examples into cached file {}".format(cached_sequentialized_single_raw_examples_file))
            torch.save(sequentialized_single_raw_examples, cached_sequentialized_single_raw_examples_file)

    dataset_input_ids = torch.tensor([example.input_ids for example in sequentialized_single_raw_examples], dtype=torch.long)
    dataset_input_masks = torch.tensor([example.input_mask for example in sequentialized_single_raw_examples], dtype=torch.long)
    dataset_segment_ids = torch.tensor([example.segment_ids for example in sequentialized_single_raw_examples], dtype=torch.long)
    dataset_label_ids = torch.tensor([example.label_ids for example in sequentialized_single_raw_examples], dtype=torch.long)
    dataset_evaluate_label_ids = [example.evaluate_label_ids for example in sequentialized_single_raw_examples]
    tensor_dataset = TensorDataset(dataset_input_ids, dataset_input_masks, dataset_segment_ids, dataset_label_ids)

    return tensor_dataset, dataset_evaluate_label_ids


def main(args_namespace):
    args = args_namespace
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: False",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in dataset_to_single_raw_example_list_processors:
        raise ValueError("Task not found: %s" % args.task_name)

    dataset_to_single_raw_example_list_processor = dataset_to_single_raw_example_list_processors[args.task_name]()

    label_list = dataset_to_single_raw_example_list_processor.get_all_possible_labels(args.tagging_schema)
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else './cache/bert-base-chinese-config.json',
        num_labels=num_labels,
        finetuning_task=args.task_name
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir='./cache'
    )

    config.absa_type = args.absa_type
    config.tfm_mode = args.tfm_mode
    config.fix_tfm = args.fix_tfm

    # gaussian
    # config.gaussian = args.gaussian
    config.per_gpu_train_batch_size = args.per_gpu_train_batch_size
    config.per_gpu_eval_batch_size = args.per_gpu_eval_batch_size
    config.max_seq_len = 231

    model = model_class.from_pretrained(
        config=config,
        pretrained_model_name_or_path=args.model_name_or_path,
        cache_dir='./cache'
    )

    model.to(args.device)

    if args.local_rank != -1:  # 分布式模型
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        pass
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        logger.info("model.device_ids: %s" % str(model.device_ids))

    if args.do_train:
        config.mode = 'train'
        train_dataset, train_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode='train')
        global_batch_load_counter, average_loss_per_batch = train(args, train_dataset, model, tokenizer)

    results = {}
    best_f1 = -99999.0
    best_checkpoint = None
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=False)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)

    logger.info("Perform validation on the following checkpoints: %s", checkpoints)
    test_results = {}
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        if global_step == 'finetune' or global_step == 'train' or global_step == 'fix' or global_step == 'overfit':
            continue

        logger.info("  loading checkpoint model %s" % checkpoint)
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        config.mode = 'dev'
        dev_result = evaluate(args, model, tokenizer, mode='dev', prefix=global_step)

        # regard the micro-f1 as the criteria of model selection
        if int(global_step) > 1000 and dev_result['micro-f1'] > best_f1:
            best_f1 = dev_result['micro-f1']
            best_checkpoint = checkpoint

        dev_result = dict((k + '_{}'.format(global_step), v) for k, v in
                          dev_result.items())

        results.update(dev_result)

        test_result = evaluate(args, model, tokenizer, mode='test', prefix=global_step)
        test_result = dict((k + '_{}'.format(global_step), v) for k, v in
                           test_result.items())
        test_results.update(test_result)

    best_ckpt_string = "\nThe best checkpoint is %s" % best_checkpoint
    logger.info(best_ckpt_string)

    dev_f1_values, dev_loss_values = [], []

    for k in results:
        v = results[k]
        if 'micro-f1' in k:
            dev_f1_values.append((k, v))
        if 'batch_average_loss' in k:
            dev_loss_values.append((k, v))

    test_f1_values, test_loss_values = [], []
    for k in test_results:
        v = test_results[k]
        if 'micro-f1' in k:
            test_f1_values.append((k, v))
        if 'batch_average_loss' in k:
            test_loss_values.append((k, v))

    log_file_path = '%s/log.txt' % args.output_dir
    log_file = open(log_file_path, 'a')
    log_file.write("\tValidation:\n")
    for (test_f1_k, test_f1_v), (test_loss_k, test_loss_v), (dev_f1_k, dev_f1_v), (dev_loss_k, dev_loss_v) in zip(
            test_f1_values, test_loss_values, dev_f1_values, dev_loss_values):  # 总共循环 15 次，对于每一个 checkpoint 均作一次。
        global_step = int(test_f1_k.split('_')[-1])
        if not args.overfit and global_step <= 1000:  # 略过 global_step <= 1000 的 checkpoint，即较早生成的 checkpoint
            continue

        print('test-%s: %.4lf, test-%s: %.4lf, dev-%s: %.4lf, dev-%s: %.4lf' % (
            test_f1_k, test_f1_v,
            test_loss_k, test_loss_v,
            dev_f1_k, dev_f1_v,
            dev_loss_k, dev_loss_v))

        validation_string = '\t\tdev-%s: %.4lf, dev-%s: %.4lf' % (dev_f1_k, dev_f1_v, dev_loss_k, dev_loss_v)
        log_file.write(validation_string + '\n')



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    seed_numbers = [593]
    learning_rates = [5e-05]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    for run_id, seed in enumerate(seed_numbers):
        start_time = datetime.datetime.now()
        for lr in learning_rates:

            logger.info("run_id: %d" % run_id)
            logger.info("seed: %d" % seed)
            logger.info("learning_rate: %f" % lr)

            log_file = 'log.txt'

            if run_id == 0 and os.path.exists(log_file):
                os.remove(log_file)
            with open(log_file, 'a') as file:
                file.write("\nIn run %s/5 (seed %s):\n" % (run_id, seed))

            args_dict = {
                'data_dir': './data/auto_home/data_IOE/normal',
                'model_type': 'bert',
                'absa_type': 'lstm',
                'tfm_mode': 'finetune',
                'fix_tfm': 0,
                'model_name_or_path': 'bert-base-chinese',
                'task_name': 'auto_home',
                'config_name': '',
                'tokenizer_name': '',
                'cache_dir': '',
                'max_seq_length': 231,
                'do_train': True,
                'do_eval': True,
                'evaluate_during_training': False,
                'do_lower_case': True,
                'per_gpu_train_batch_size': 16,
                'per_gpu_eval_batch_size': 16,
                'gradient_accumulation_steps': 1,
                'learning_rate': lr,
                'weight_decay': 0.0,
                'adam_epsilon': 1e-08,
                'max_grad_norm': 1.0,
                'num_train_epochs': 2.0,
                'max_steps': 5000,
                'warmup_steps': 500,
                'logging_steps': 50,
                'save_steps': 100,
                'eval_all_checkpoints': True,
                'no_cuda': False,
                'overwrite_output_dir': True,
                'overwrite_cache': False,
                'seed': seed,
                'tagging_schema': 'BIEOS',
                'overfit': 0,
                'local_rank': -1,
                'output_dir': 'bert-tfm-auto_home-finetune'
            }

            args_namespace = argparse.Namespace(**args_dict)

            main(args_namespace)

            end_time = datetime.datetime.now()
            duration = (end_time - start_time).seconds

            logger.info("run_id %s duration time: %s" % (str(run_id), str(datetime.timedelta(seconds=duration))))
