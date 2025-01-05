import os
import re
import pandas as pd

def process_m8_file(file_path, n_prot):
    # 创建一个空字典来存储每种查询蛋白质的最大相似度
    max_similarity = {}

    # 读取 m8 文件
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            query_id = parts[0]
            match_id = parts[1]
            similarity = float(parts[2])

            # 更新最大相似度
            if query_id not in max_similarity:
                max_similarity[query_id] = similarity
            else:
                max_similarity[query_id] = max(max_similarity[query_id], similarity)
    
    novel_sum = 0
    for sim in max_similarity.values():
        novel_sum += 1 - sim

    novelty = (novel_sum + (n_prot - len(max_similarity)) * 1) / n_prot
    
    return round(novelty, 3)

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

def main(folder_path, output_csv):
    results = []

    # 遍历指定文件夹中的所有 .m8 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.m8'):
            file_path = os.path.join(folder_path, filename)
            novelty = process_m8_file(file_path, extract_num(filename))
            results.append((filename, novelty))

    # 将结果保存到 CSV 文件中
    df = pd.DataFrame(results, columns=['File', 'novelty'])
    df.to_csv(output_csv, index=False)

import sys

if __name__ == "__main__":
    folder_path = sys.argv[1]
    output_csv = sys.argv[2]
    
    main(folder_path, output_csv)