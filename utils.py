import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np
import random
import argparse
import pandas as pd
import math
import os
DEFAULT_MAX_SEQ_LEN = 512

def init_tokenizer(tokenizer):
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        # print(param_group['params'])
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_seed(seed):
    torch.manual_seed(seed + get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def info_nce_loss():
    pass



def build_paag_args():
    parser = argparse.ArgumentParser(description='protein')
    parser.add_argument("--SSL_loss", type=str, default="InfoNCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--representation_frozen", dest='representation_frozen', action='store_true')
    parser.add_argument('--no_representation_frozen', dest='representation_frozen', action='store_false')
    parser.set_defaults(representation_frozen=False)
    parser.add_argument("--output_model_dir", type=str, default='../saved_model/')
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--if_saved_model", type=bool, default=False)
    parser.add_argument("--taiji", type=bool, default=False)
    parser.add_argument("--devcloud", type=bool, default=False)
    # args = parser.parse_args()

    parser.add_argument('--config', default='configs/pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--name', default='paag', type=str)
    args = parser.parse_args()
    args.output_model_dir = '../saved_model'
    return args

def freeze_network(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def primary2sequence():
    primary_accession = pd.read_json(path_or_buf="data/uniprotkb_pairs.jsonl", lines=True).replace('\n', '', regex=True)
    library = pd.read_json(path_or_buf="data/uniprotkb.jsonl", lines=True)
    primary_accession["sequence"] = ""
    pd.set_option('max_colwidth', 1000)
    pd.set_option('display.max_columns', None)
    for index, row in primary_accession.iterrows():
        seq = library.loc[library["primaryAccession"]==row["primaryAccession"]]["sequence"]
        primary_accession.loc[index, 'sequence'] = dict(list(dict(seq).values())[0])['value']
    output_path = "/data/pairs.jsonl"
    pairs = primary_accession.to_json(orient='records', lines=True)
    with open(output_path, 'w') as f:
        f.write(pairs)


class GPTconfig:
    resid_drop = 0.1
    attn_drop = 0.1
    pos_drop = 0.1
    block_size = 512
    vocab_size = 65

    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

class TrainingConfig:
    max_epochs = 100
    lr = 3e-4
    betas = (0.9,0.95)
    weight_decay = 0.1
    epsilon = 10e-8
    batch_size = 64
    grad_norm_clip = 1.0
    lr_decay = True
    num_workers = 8
    warmup_tokens = 375e6
    final_tokens = 260e9
    shuffle = True
    pin_memory = True
    device = "cuda"
    ckpt_path = "./transformers.pt"

    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)


if __name__ == '__main__':
    primary2sequence()