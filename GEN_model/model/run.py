import sys
import os
import warnings
import argparse
import fitlog
import torch
from data.pipe import BartBPEABSAPipe
from model.bart import BartSeq2SeqModel, Restricter
from fastNLP import Trainer, CrossEntropyLoss, Tester
from model.metrics import Seq2SeqSpanMetric
from model.losses import Seq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback, SequentialSampler, RandomSampler
from model.callbacks import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from model.generater import SequenceGeneratorModel

sys.path.append('../')

if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']

warnings.filterwarnings('ignore')
fitlog.set_log_dir('logs')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='auto_home/data_for_BART', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score', 'avg_feature'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)                      # transformers == 3.4.0, pytorch == 1.7.1，可以自动下载 bart-base
# parser.add_argument('--bart_name', default="../bart/facebook-bart-base/", type=str)
# parser.add_argument('--bart_name', default='fnlp/bart-base-chinese', type=str)                # 远程加载 bart-base-chinese
# parser.add_argument('--bart_name', default="../bart/fnlp-bart-base-chinese/", type=str)         # 加载本地的 bart-base-chinese
parser.add_argument('--use_encoder_mlp', type=int, default=1)

args = parser.parse_args()
lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
fitlog.add_hyper(args)
use_encoder_mlp = args.use_encoder_mlp

demo = False
bart_dir = None
if demo:
    if bart_name == "facebook/bart-base":
        bart_dir = "facebook_bart_base_remote"
    elif bart_name == "../bart/facebook-bart-base/":
        bart_dir = "facebook_bart_base_local"
    elif bart_name == "fnlp/bart-base-chinese":
        bart_dir = "fnlp_bart_base_chinese_remote"
    elif bart_name == "../bart/fnlp-bart-base-chinese/":
        bart_dir = "fnlp_bart_base_chinese_local"
    cache_fn = f"caches/data_{bart_dir}_{dataset_name}_{opinion_first}_demo.pt"
else:
    if bart_name == "facebook/bart-base":
        bart_dir = "facebook_bart_base_remote"
    elif bart_name == "../bart/facebook-bart-base/":
        bart_dir = "facebook_bart_base_local"
    elif bart_name == "fnlp/bart-base-chinese":
        bart_dir = "fnlp_bart_base_chinese_remote"
    elif bart_name == "../bart/fnlp-bart-base-chinese/":
        bart_dir = "fnlp_bart_base_chinese_local"
    cache_fn = f"caches/data_{bart_dir}_{dataset_name}_{opinion_first}.pt"

@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartBPEABSAPipe(tokenizer_path=bart_name, opinion_first=opinion_first)
    data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id


data_bundle, tokenizer, mapping2id = get_data()

max_len = 10
max_len_a = {
    'fan/14lap': 0.5,
    'fan/14res': 0.7,
    'fan/15res': 0.4,
    'fan/16res': 0.6,
    'auto_home/data_for_BART': 0.7
}[dataset_name]

bos_token_id = 0                            # 在 tokenizer 中:
                                            #   tokenizer.bos_token_id:   0
                                            #   tokenizer.bos_token:      <s>
eos_token_id = 1                            # 在 tokenizer 中:
                                            #   tokenizer.eos_token_id:   2
                                            #   tokenizer.eos_token:      </s>

label_ids = list(mapping2id.values())       # mapping2id: {'tag': 50265}
                                            #   真正加入 vocab 的 token 是: <<tag>>

model = BartSeq2SeqModel.build_model(
    bart_name,                              # 'facebook/bart-base'
    tokenizer,
    label_ids=label_ids,
    decoder_type=decoder_type,
    copy_gate=False,
    use_encoder_mlp=use_encoder_mlp,
    use_recur_pos=False
)

vocab_size = len(tokenizer)
restricter = Restricter(label_ids)

model = SequenceGeneratorModel(
    model,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    max_length=max_len,
    max_len_a=max_len_a,
    num_beams=num_beams,
    do_sample=False,
    repetition_penalty=1,
    length_penalty=length_penalty,
    pad_token_id=eos_token_id,
    restricter=None
)

if torch.cuda.is_available():
    if 'p' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        device = 'cuda'
    else:
        device = 'cuda'
else:
    device = 'cpu'

parameters = []
params = {'lr': lr, 'weight_decay': 1e-2}
params['params'] = [param for name, param in model.named_parameters() if
                    not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr': lr, 'weight_decay': 1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr': lr, 'weight_decay': 0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
if 'dev' in data_bundle.datasets:
    callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))
    dev_data = data_bundle.get_dataset('dev')
else:
    callbacks.append(FitlogCallback())
    dev_data = data_bundle.get_dataset('test')

sampler = RandomSampler()
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1,
                  dev_data=dev_data, metrics=metric, metric_key='oe_f',
                  validate_every=-1, save_path=None, use_tqdm='SEARCH_ID' not in os.environ, device=device,
                  callbacks=callbacks, check_code_level=-1 if 'SEARCH_ID' in os.environ else 0, test_use_tqdm=False,
                  test_sampler=SequentialSampler(), dev_batch_size=batch_size * 5)

trainer.train(load_best_model=False)

if trainer.save_path is not None:
    model_name = "best_" + "_".join([model.__class__.__name__, trainer.metric_key, trainer.start_time])
    fitlog.add_other(name='model_name', value=model_name)
