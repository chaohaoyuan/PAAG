# diversity calculation

1. `cd diversity_search`

2. move fasta files into folder `query`

3. `bash diversity_match.sh &> xxx.log`

This command will generate .m8 files corresponding to the query fasta in the folder `./diversity_search/result`.

4. `python mean_diversity_cal.py ./diversity_search/result ./diversity_results.csv`

Then we can find the diversity of each fasta in the file `diversity_results.csv`.