import os
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import BasePredictionWriter
from models.paag_pretrain_pl import PAAG_Pretrain
from dataset import TextSeqPair
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import re
seed_everything(42)
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


def customBatchBuilder(samples):
    key, seq, text, domain = zip(*samples)
    seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
    return key, seq, text, domain


def customBatchBuilder2(samples):
    seq, text = zip(*samples)
    seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
    return seq, text


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        file = pd.read_json(path_or_buf=self.output_dir, lines=True)
        generated = pd.DataFrame(predictions, columns=['text', 'sequence'])
        df = pd.concat([file, generated])
        pairs = df.to_json(orient='records', lines=True)
        with open(self.output_dir, 'w') as f:
            f.write(pairs)

if __name__ == '__main__':
    output_path = "./data/paag_generated_data.jsonl"

    dataset_test = pd.read_json(path_or_buf="data/test_ig_zinc.jsonl", lines=True)
    dataset_test = TextSeqPair(dataset_test)

    batch_size = 1
    test_dataloader = DataLoader(dataset_test, num_workers=16, batch_size=batch_size, shuffle=False,
                                 collate_fn=customBatchBuilder2,
                                 drop_last=True)

    with open(output_path, "w") as file:
        file.seek(0)
        file.truncate()  # delete
        file.close()

    pred_writer = CustomWriter(output_dir=output_path, write_interval="epoch")

    output_path = "../weights/paag.ckpt"
    model = PAAG_Pretrain.load_from_checkpoint(
        output_path,
        map_location="cpu",
        strict=False
        )
    model.eval()

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp',
        devices=8,
        num_nodes=1,
        callbacks=[pred_writer],
        precision="bf16-mixed",
    )
    trainer.predict(model, test_dataloader)
