import os
import csv
from collections import defaultdict

def process_m8_file(file_path, num_total):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    protein_similarities = defaultdict(list)

    for line in lines:
        if line.strip():
            columns = line.split('\t')
            query_protein_id = columns[0]
            subject_protein_id = columns[1]
            similarity = float(columns[2])

            if query_protein_id != subject_protein_id:
                protein_similarities[query_protein_id].append(1.0 - similarity)

    average_similarity = 0
    sum_sim_match = 0
    for values in protein_similarities.values():
        sum_sim_match += (sum(values) + num_total - 1 - len(values)) / (num_total - 1)
    sum_sim_no_match = num_total - len(protein_similarities)
    average_similarity = (sum_sim_match + sum_sim_no_match) / num_total
    return round(average_similarity, 3), len(protein_similarities)

import re

def extract_num(input_string):
    # 使用正则表达式提取'_'后的数字
    numbers = re.findall(r'_(\d+)', input_string)

    # 如果找到了数字
    if numbers:
        # 将第一个找到的数字转换为整数并返回
        return int(numbers[-1])
    else:
        print("No numbers found in the input string.")
        return None

import sys

def main():
    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file', 'mean_diversity', 'num_protein_match'])

        for file in os.listdir(input_folder):
            if file.endswith('.m8'):
                file_path = os.path.join(input_folder, file)
                average_similarity, num_query_proteins = process_m8_file(file_path, extract_num(file[:-3]))
                csv_writer.writerow([file[:-3], average_similarity, num_query_proteins])

if __name__ == '__main__':
    main()