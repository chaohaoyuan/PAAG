#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_fasta_filename>"
    exit 1
fi

input_file="$1"
base_name=$(basename "$input_file" .fasta)

e_values=(100 0.001 0.01 0.1 1 10)

for e in "${e_values[@]}"; do
    pfam_scan.pl -fasta ./"$base_name".fasta -dir ./pfamdb -outfile ./"$e"_"$base_name".fa -e_dom "$e" -e_seq "$e"
done
