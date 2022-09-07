import argparse
import copy, json, os
import datetime
from collections import Counter
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from transformers import BertConfig
from transformers.optimization import AdamW, WarmupLinearSchedule
from model.model import BiDAF, BertBiDAF
from model.data import SQuAD
from model.ema import EMA
import evaluate
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def train(args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    bert_config = BertConfig()
    bert_config.tfm_mode = 'finetune'

    model = BertBiDAF(args=args, bert_config=bert_config).to(device)
    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.is_leaf is not True:
            print('param:', param)
            print('name:', name)
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = AdamW(parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.max_steps * args.warmup_proportion), t_total=args.max_steps)

    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_micro_f1 = -1

    iterator = data.train_iter

    batch_iterator = tqdm(
        iterator,
        desc="\tBatch (Iteration or Step)",
        position=0,
    )

    for i, batch in enumerate(batch_iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('\nepoch:{}\t{}'.format(present_epoch + 1, datetime.datetime.now()))
        last_epoch = present_epoch
        start_idx_logits, end_idx_logits = model(batch)
        optimizer.zero_grad()
        batch_loss = criterion(start_idx_logits, batch.s_idx) + criterion(end_idx_logits, batch.e_idx)
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)

        if (i + 1) % args.print_freq == 0:
            dev_loss, macro_f1, micro_precision, micro_recall, micro_f1 = test(model, ema, args, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('results/macro_f1', macro_f1, c)
            writer.add_scalar('results/micro_precision', micro_precision, c)
            writer.add_scalar('results/micro_recall', micro_recall, c)
            writer.add_scalar('results/micro_f1', micro_f1, c)

            print(f'\ttrain loss: {loss:.4f} / '
                  f'dev loss: {dev_loss:.4f}'f' / '
                  f'dev macro_f1: {macro_f1:.4f} / '
                  f'dev micro_f1: {micro_f1:.4f}')

            if micro_f1 > max_dev_micro_f1:
                max_dev_micro_f1 = micro_f1
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()
    writer.close()
    print("\nmax dev micro_f1: ".format(max_dev_micro_f1))

    return best_model


def test(model, ema, args, data):
    all_b_e_s_pred_triplets = []
    device = torch.device("cuda:{}".format(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    model.eval()

    aspect_ch_en = {
        "内": "NS",
        "动": "DL",
        "外": "WG",
        "性": "XJB",
        "操": "CK",
        "空": "KJ",
        "能": "NH",
        "舒": "SSX"
    }

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    with torch.set_grad_enabled(False):
        for batch_index, batch in enumerate(iter(data.dev_iter)):

            # 1. 首先取得当前 batch 所包含的 aspect
            batch_aspect_tensors = batch.q_word[0].data
            batch_aspect_first_tokens = []
            batch_aspects = []
            for t in batch_aspect_tensors:
                aspect_first_token = tokenizer.convert_ids_to_tokens(t.tolist())[1]
                batch_aspect_first_tokens.append(aspect_first_token)
            for token in batch_aspect_first_tokens:
                batch_aspects.append(aspect_ch_en[token])

            p1, p2 = model(batch)

            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for example_index in range(batch_size):

                b = s_idx[example_index].item()
                e = e_idx[example_index].item()
                aspect = batch_aspects[example_index]
                triplet = [(b, e, aspect)]

                all_b_e_s_pred_triplets.append(triplet)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))


    results = evaluate.main(args, all_b_e_s_pred_triplets=all_b_e_s_pred_triplets)
    return loss, results['macro_f1'], results['micro_precision'], results['micro_recall'], results['micro_f1']


def test_original(model, ema, args, data):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    with torch.set_grad_enabled(False):

        for batch_index, batch in enumerate(iter(data.dev_iter)):
            p1, p2 = model(batch)
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for example_index in range(batch_size):
                id = batch.id[example_index]
                answer = batch.c_word[0][example_index][s_idx[example_index]: e_idx[example_index] + 1]
                answer = ''.join([tokenizer.convert_ids_to_tokens(idx.item()) for idx in answer])
                answers[id] = answer

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers, ensure_ascii=False), file=f)

    results = evaluate.main(args)
    return loss, results['exact_match'], results['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=2, type=int)
    # parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dev-file', default='dev-v0.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=14, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=768, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    # parser.add_argument('--learning-rate', default=0.5, type=float)
    # parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--print-freq', default=200, type=int)
    parser.add_argument('--train-batch-size', default=8, type=int)
    # parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--train-file', default='train-v0.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    # parser.add_argument('--num_gpu', default=2, type=int)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    args = parser.parse_args()

    learning_rates = [
        5e-5,
        ]

    for lr in learning_rates:
        print("current learning rate: {}".format(lr))
        args.learning_rate = lr
        data = SQuAD(args)

        setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
        setattr(args, 'dataset_file', f'.data/data_IOE/normal/{args.dev_file}')
        setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
        setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
        print('data loading complete!')

        print('training start!')
        best_model = train(args, data)
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        torch.save(best_model.state_dict(), f'saved_models/BiDAF_{args.model_time}.pt')
        print('training finished!')


if __name__ == '__main__':
    main()
