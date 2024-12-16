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
from dataset import TextSeqPair
from models.paag_pretrain_pl import paag_pretrain_protst
import utils
from utils import build_paag_args, freeze_network
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pytorch_lightning import seed_everything
current_directory = Path(__file__).parent.absolute()
seed_everything(42)


def main(args, config):
    os.environ['CURL_CA_BUNDLE'] = ''
    wandb_logger = WandbLogger(project="PAAG",
                               name=config['name'],
                               group=config['name'],
                               save_dir=current_directory,
                               config=args,
                               )
    def customBatchBuilder(samples):
        seq, text = zip(*samples)
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
        return seq, text

    dataset = pd.read_json(path_or_buf="data/protst_text.jsonl", lines=True)
    train_dataset = TextSeqPair(dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=16, batch_size=config['batch_size'], shuffle=False,
                                  collate_fn=customBatchBuilder,
                                  drop_last=True, pin_memory=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_model_dir,
        filename=config['name'] + '{epoch:02d}',
        save_top_k=-1
    )

    model = paag_pretrain_protst(queue_size=config['queue_size'], model_name_seq=config['model'], seq_config=config['seq_config'])
    freeze_network(model.text_encoder)
    trainer = pl.Trainer(
        default_root_dir=current_directory,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=8,
        gradient_clip_val=1,
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