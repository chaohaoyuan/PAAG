#!/bin/bash

FASTA_FOLDER="./query"
RESULT_FOLDER="./result"
DB_path='./DB/uniprotdb'

# 创建结果文件夹（如不存在则创建）
mkdir -p "$RESULT_FOLDER"
mkdir -p tmp

rm -rf ${RESULT_FOLDER}/*

# 遍历每个 fasta 文件
for query_file in "$FASTA_FOLDER"/*.fasta; do

    # 清空 tmp 文件夹
    rm -rf tmp/*

    # 待匹配的 fasta 文件名
    fasta_name=$(basename "$query_file" .fasta)

    # 匹配结果文件路径
    result_file="$RESULT_FOLDER/${fasta_name}.m8"

    # 运行 mmseqs 命令
    mmseqs easy-search ${query_file} ${DB_path} ${result_file} tmp -e 100

    echo "处理文件: $fasta_name, 结果保存于: $result_file"
done

echo "所有文件处理完成！"
