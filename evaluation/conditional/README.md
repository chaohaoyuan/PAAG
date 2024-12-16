We employ Pfam to evaluate the success rate of the conditional generated proteins.

Here, we first to introduce how to set up Pfam dataset from the totorial: https://github.com/aziele/pfam_scan

Then, we also need to install the pfam_scan from conda.

After these preparation, we can use the bash under this folder `./run_pfam_scan.sh file_name` to search all the possible functional domains inside the given proteins. 

To test whether the proteins contain zinc-finger or ig domains, please refer to `analyze.py`. You can also try to custom any other functional domains to fulfill your own requirement.

## Dataset Preparation

1. Download two files from the Pfam FTP site:

```
wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz
wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
```

2. Unpack the downloaded files to the same directory.

```
mkdir pfamdb
gunzip -c Pfam-A.hmm.dat.gz > pfamdb/Pfam-A.hmm.dat
gunzip -c Pfam-A.hmm.gz > pfamdb/Pfam-A.hmm
rm Pfam-A.hmm.gz Pfam-A.hmm.dat.gz
```

3. Prepare Pfam database for HMMER by creating binary files.

```
hmmpress pfamdb/Pfam-A.hmm
```

## Installation

One can install pfam_scan also from the tutorial: https://github.com/aziele/pfam_scan

Here we provide another simple alternative from conda: https://anaconda.org/bioconda/pfam_scan

```
conda install bioconda::pfam_scan
```