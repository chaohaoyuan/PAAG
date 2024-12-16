import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import json
from pathlib import Path
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from dataset import Uniprot_LocationDataset
from models.paag_pretrain_pl import paag_pretrain, paag_pretrain_partial_local
import utils
from utils import build_paag_args, freeze_network
import wandb
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from pytorch_lightning import seed_everything
current_directory = Path(__file__).parent.absolute()
seed_everything(42)



def extract_subsequences_domains(seq, data):
    subsequences = []
    domains = []
    unsubsequences = []
    if len(data) == 0:
        return [], [], []
    for i, item in enumerate(data):
        if i == 2:
            break
        if item['location'][0] == None or item['location'][1] == None:
            continue
        sub_length = item['location'][1] - item['location'][0] + 1
        remains = seq[:item['location'][0]] + seq[item['location'][1]:]
        start = list(range(len(remains)))
        if len(start) < max(2, sub_length/2):
            continue
        start_index = random.sample(start, 3)
        subsequences.append(" ".join(list(re.sub(r"[UZOB]", "X", seq[item['location'][0]: item['location'][1] + 1]))))
        unsubsequences.append(
            " ".join(list(re.sub(r"[UZOB]", "X", remains[start_index[0]: max(len(remains), (start_index[0] + sub_length))]))))
        unsubsequences.append(
            " ".join(list(re.sub(r"[UZOB]", "X", remains[start_index[1]: max(len(remains), (start_index[1] + sub_length))]))))
        unsubsequences.append(
            " ".join(list(re.sub(r"[UZOB]", "X", remains[start_index[2]: max(len(remains), (start_index[2] + sub_length))]))))
        domains.append(item['textural_description'])
    return subsequences, unsubsequences, domains


def main(args, config):
    os.environ['CURL_CA_BUNDLE'] = ''
    wandb_logger = WandbLogger(project="PAAG",
                               name=config['name'],
                               group=config['name'],
                               save_dir=current_directory,
                               settings=wandb.Settings(start_method="fork"),
                               config=args,
                               )

    dataset = pd.read_json(path_or_buf="data/toy_dataset.jsonl", lines=True)
    dataset = Uniprot_LocationDataset(dataset)

    if config['match'] == 'partial_local_align':
        model = paag_pretrain_partial_local(queue_size=config['queue_size'], model_name_seq=config['model'], seq_config=config['seq_config'])
        def customBatchBuilder(samples):
            seq, text, functionDomainRange = zip(*samples)
            subsequences = []
            noisy_subsequences = []
            domains = []
            for i in range(len(seq)):
                subsequence, noisy_subsequence, domain = extract_subsequences_domains(seq[i], functionDomainRange[i])
                subsequences.append(subsequence)
                noisy_subsequences.append(noisy_subsequence)
                domains.append(domain)
            seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
            return seq, text, subsequences, noisy_subsequences, domains
    elif config['dataset'] == 'uniprot_location':
        model = paag_pretrain()

        def customBatchBuilder(samples):
            seq, text, functionDomainRange = zip(*samples)
            seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
            return seq, text


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_model_dir,
        filename=config['name'] + '{epoch:02d}',
        save_top_k=-1
    )


    print(len(dataset))
    lr_logger = pl.callbacks.LearningRateMonitor()
    train_dataloader = DataLoader(dataset, num_workers=16, batch_size=1, shuffle=False,
                                 drop_last=False, collate_fn=customBatchBuilder)

    trainer = pl.Trainer(
        default_root_dir=current_directory,
        logger=wandb_logger,
        callbacks=[lr_logger, checkpoint_callback],
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=8,
        gradient_clip_val=0.25,
        log_every_n_steps=10,
        max_epochs=config['max_epoch'],
        num_nodes=2,
        precision="bf16-mixed",
    )

    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    args = build_paag_args()
    print(os.getcwd())
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args, config)