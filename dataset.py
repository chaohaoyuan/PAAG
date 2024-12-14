import json
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pandas as pd
import csv
from collections import defaultdict
import re
import os
from Bio import SeqIO



class TextSeqPair(Dataset):
    def __init__(self, json_file):
        self.json_file = json_file

    def __getitem__(self, idx):
        return self.json_file.iloc[idx, 1], self.json_file.iloc[idx, 0]

    def __len__(self):
        return len(self.json_file)



class Uniprot_LocationDataset(Dataset):
    def __init__(self, json_file):
        self.json_file = json_file

    def __getitem__(self, idx):
        # print(self.json_file.iloc[idx, ])
        return self.json_file.iloc[idx, 4]['value'], self.json_file.iloc[idx, 1], self.json_file.iloc[idx, 2] # seq, text, location

    def __len__(self):
        return len(self.json_file)

