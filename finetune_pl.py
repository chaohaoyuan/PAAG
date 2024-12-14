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
from models.blip_pretrain_pl import PAAG_Pretrain_Partial_Local
import utils
from utils import build_blip_args
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
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
                               settings=wandb.Settings(start_method="fork"),
                               config=args,
                               )


    if config['dataset'] == 'deeploc':
        dataset = pd.read_json(
            path_or_buf="/mnt/test/buddy1/chaohaoyuan/protein/data/DeepLoc_v1/DeepLoc_v1_train.jsonl",
            lines=True)
        train_dataset = Uniprot_LocationDataset(dataset)

    def customBatchBuilder(samples):
        seq, text, functionDomainRange = zip(*samples)
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
        return seq, text


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_model_dir,
        filename=config['name'] + '{epoch:02d}',
        save_top_k=-1,
        every_n_epochs = 5,
    )

    print(len(train_dataset))
    # train_dataset = dataset
    if config['corrupt_strategy'] == 'aug_X_not_decoder':
        train_dataset = NoisyDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=16, batch_size=config['batch_size'], shuffle=False,
                                  collate_fn=customBatchBuilder,
                                  drop_last=True)

    model = PAAG_Pretrain_Partial_Local.load_from_checkpoint(
        'weights/paag_protbert_protannotation.ckpt',
        map_location="cpu",
    )
    model.queue_size = 4096
    model.seq_queue = model.seq_queue[:, :model.queue_size]
    model.text_queue = model.text_queue[:, :model.queue_size]
    model.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    print('re-organize queue = 4096')

    lr_logger = pl.callbacks.LearningRateMonitor()


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
        num_nodes=1,
        precision="bf16-mixed",
    )

    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    args = build_blip_args()
    print(os.getcwd())
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)




    main(args, config)