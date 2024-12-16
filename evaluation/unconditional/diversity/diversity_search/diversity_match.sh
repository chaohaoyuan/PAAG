#!/bin/bash

FASTA_FOLDER="./query"
RESULT_FOLDER="./result"

mkdir -p "$RESULT_FOLDER"
mkdir -p tmp

rm -rf ${RESULT_FOLDER}/*

for query_file in "$FASTA_FOLDER"/*.fasta; do

    rm -rf tmp/*

    fasta_name=$(basename "$query_file" .fasta)

    result_file="$RESULT_FOLDER/${fasta_name}.m8"

    mmseqs easy-search ${query_file} ${query_file} ${result_file} tmp -e 1000000 -s 15

    echo "processing: $fasta_name, saving into: $result_file"
done

echo "All files have been processed successfully!"
