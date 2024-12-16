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
import os
import csv
import sys
from statistics import mean

def process_csv_files(input_folder):
    output_file = os.path.join(input_folder[:-7], 'cond_diversity.csv')
    results = []

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                non_empty_rows = 0
                second_column_values = []

                for row in reader:
                    if row:
                        non_empty_rows += 1
                        second_column_values.append(float(row[1]))

                cnt_sum = sum(second_column_values)
                total = extract_num(file[:-4]) 
                miss_rows = total- non_empty_rows
                average = (cnt_sum + miss_rows) / total
                average = round(average, 3)
                results.append((file[:-4], average, miss_rows))

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "diversity", "Miss Rows"])
        for result in results:
            writer.writerow(result)

input_folder = sys.argv[1]
process_csv_files(input_folder)