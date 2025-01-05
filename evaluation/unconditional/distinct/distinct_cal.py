import sys
import csv
import os
from collections import Counter

# 读取 FASTA 文件
def read_fasta(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    proteins = []
    protein_id = ''
    protein_seq = ''
    for line in lines:
        if line.startswith('>'):
            if protein_seq:
                proteins.append((protein_id, protein_seq))
                protein_seq = ''
            protein_id = line.strip()[1:]
        else:
            protein_seq += line.strip()
    if protein_seq:
        proteins.append((protein_id, protein_seq))
    return proteins

# 计算 n-gram 种类数
def count_ngrams(protein_seq, n):
    ngram_counter = Counter()
    for i in range(len(protein_seq) - n + 1):
        ngram = protein_seq[i:i+n]
        ngram_counter[ngram] += 1
    return len(ngram_counter)

# 计算平均值和加权平均值
def calculate_means(proteins, n):
    total_count = 0
    total_length = 0
    total_gram = 0
    for protein in proteins:
        count = count_ngrams(protein[1], n)
        length = len(protein[1])
        total_count += count
        total_length += length
        total_gram += length - n + 1
    # average_count = round(total_count / len(proteins), 2)
    # weighted_average_count = 0
    # for protein in proteins:
    #     weight = len(protein[1]) / total_length
    #     weighted_average_count += weight * count_ngrams(protein[1], n)
    # weighted_average_count = round(weighted_average_count, 2)
    
    # calculate the normalized distinct
    distinct_norm = round(total_count / total_gram, 4)

    return distinct_norm

# 主程序
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python fasta_ngram.py <fasta_folder> <n>')
        sys.exit(1)
    fasta_folder = sys.argv[1]
    n = int(sys.argv[2])
    result_file = f"distinct_result_n_{n}.csv"
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "distinct normalized"])
        for filename in os.listdir(fasta_folder):
            if filename.endswith(".fasta"):
                fasta_file = os.path.join(fasta_folder, filename)
                proteins = read_fasta(fasta_file)
                distinct_norm = calculate_means(proteins, n)
                writer.writerow([filename[:-6], distinct_norm])