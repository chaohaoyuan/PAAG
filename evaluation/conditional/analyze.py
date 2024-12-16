import pandas as pd
import re
import pickle
import numpy as np

values = []

def read_data(file_path):
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file if not line.startswith("#")]
    with open("temp_data.txt", "w") as temp_file:
        temp_file.write("\n".join(lines))
    df = pd.read_csv("temp_data.txt", sep="\s+", header=None)
    df.columns = [
        "seq_id", "alignment_start", "alignment_end", "envelope_start", "envelope_end",
        "hmm_acc", "hmm_name", "type", "hmm_start", "hmm_end", "hmm_length",
        "bit_score", "E-value", "significance", "clan"
    ]
    import os
    os.remove("temp_data.txt")
    return df


def process_group_zinc(group):
    zf_rows = group[group["hmm_name"].str.contains(r'^zf|Zf', regex=True)]
    if not zf_rows.empty:
        global sum_values
        global number_of_groups
        number_of_groups += 1
        min_evalue = zf_rows["E-value"].min()
        sum_values += min_evalue
        values.append(min_evalue)
        ids.append(zf_rows["seq_id"].tolist()[0])
        evalue.append(min_evalue)

def process_group_ig(group):
    zf_rows = group[group["hmm_name"].str.contains(r'^(I|i)g|V-set|I-set|C2-set|C1-set|Lep_receptor_Ig|Adhes-Ig_like', regex=True)]
    if not zf_rows.empty:
        global sum_values
        global number_of_groups
        number_of_groups += 1
        min_evalue = zf_rows["E-value"].min()
        sum_values += min_evalue
        values.append(min_evalue)
        ids.append(zf_rows["seq_id"].tolist()[0])
        evalue.append(min_evalue)

domain = 'zinc'

fasta_file_path = "paag_" + domain + ".fa"
path = './'
file_list = ['100_'+fasta_file_path, '10_'+fasta_file_path, '1_'+fasta_file_path, '0.1_'+fasta_file_path, '0.01_'+fasta_file_path, '0.001_'+fasta_file_path]

for fasta_file_path in file_list:
    print(fasta_file_path)
    df = read_data(fasta_file_path)
    number_of_groups = 0
    sum_values = 0
    ids = []
    evalue = []

    # 对 'seq_id' 进行分组
    grouped = df.groupby("seq_id")

    for name, group in grouped:
        if domain == 'ig':
            process_group_ig(group)
        else:
            process_group_zinc(group)

    print("success sample:", number_of_groups)
    print("sum:", sum_values)
    print("mean:", sum_values / number_of_groups)
    print(ids)
    print(evalue)
