import os
import pandas as pd
import sys

# 输入文件夹路径
input_folder = sys.argv[1]
output_folder = sys.argv[2]
# check_val = 300

# 遍历.m9文件
for file in os.listdir(input_folder):
    if file.endswith('.m8'):
        # 读取.m9文件
        df = pd.read_csv(os.path.join(input_folder, file), sep='\t', header=None)

        # 对于每条蛋白质序列，找到除自身之外最相似的蛋白质
        df = df[df[0] != df[1]]
        grouped = df.groupby(0)
        max_similarity = grouped[2].idxmax()
        most_similar = df.loc[max_similarity]

        # 计算1.0 - 该相似度并保留三位小数
        most_similar['similarity_diff'] = (1.0 - most_similar[2]).round(3)

        # 仅保存查询蛋白质序列ID和similarity_diff
        most_similar = most_similar[[0, 'similarity_diff']]

        # 将结果存储到csv文件中（不包含表头）
        output_file = os.path.join(output_folder, os.path.splitext(file)[0] + '.csv')
        most_similar.to_csv(output_file, index=False, header=False)

        # 检查csv文件的行数是否等于300
        # if most_similar.shape[0] == check_val:
        #     print(os.path.splitext(file)[0] + ": PASS")
        # else:
        #     print(os.path.splitext(file)[0] + ": NO!!!!!!!!!!!")