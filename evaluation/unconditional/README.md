# Installation
To install MMseqs2 for evaluation, please refer to [official repository of  MMseqs2](https://github.com/soedinglab/MMseqs2.)

# Distinct calculation
Put all fastas whose distinct need to be calculated into "./distinct/fasta" directory
Enter the distinct folder
```
cd ./distinct
```
then run the distinct evaluation code
```
python ./distinct_cal.py <fasta_folder> <n>
```
- \<fasta_folder\> denotes the folder containing fastas to be evaluated. Usually "./fasta".
- \<n\> denotes calculating n-gram distinct.

then a csv file containing the distinct results of fastas will be generated under the distinct folder.

# Novelty calculation
Enter the novelty folder
```
cd ./novelty
```
Download uniprotkb fasta files
```
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.fasta.gz
wget https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_trembl.fasta.gz
gunzip xxx
```
Merge reviewed and reviewed fasta files
```
cat uniprot_sprot.fasta uniprot_trembl.fasta > uniprot.fasta
```
Create database of uniprotkb
```
mmseqs createdb uniprot.fasta novelty_search/DB/uniprotdb
```
Create database index of uniprotkb
```
mmseqs createindex novelty_search/DB/uniprotdb tmp
```
Then enter novelty_search folder
```
cd ./novelty_search
```
Put all fastas whose novelty need to be calculated into "./query" directory
Then run the novelty match script
```
bash novelty_match.sh
```
This will produce .m8 files in "./result" directory corresponding to all fastas files in the "./query" directory
Then run the novelty calculation code
```
cd ..
python novelty_cal.py ./novelty_search/result <csv_file_path>
```
\<csv_file_path\>: the result csv file path. Usually "./novelty_result.csv".

This will generate a csv file containing the novelty results of fasta files.

# Diversity calculation
```
cd ./diversity/diversity_search
```
Put all fastas whose diversity need to be calculated into "./query" directory
Then run the diversity match script
```
bash diversity_match.sh
```
This will produce .m8 files in "./result" directory corresponding to all fastas files in the "./query" directory
Then run the diversity calculation code
```
cd ..
python diversity_cal.py ./diversity_search/result <csv_file_path>
```
\<csv_file_path\>: the result csv file path. Usually "./diversity_result.csv".

This will generate a csv file containing the diversity results of fasta files.