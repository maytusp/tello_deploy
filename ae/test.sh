#!/bin/bash --login
#$ -cwd             # Job will run from the current directory
                    # NO -V line - we load modulefiles in the jobscript
#$ -l a100=1 # We request only 1 GPU
#$ -l cu122
#$ -pe smp.pe 8 # We want a CPU core for each process (see below)


module load apps/binapps/anaconda3/2021.11  # Python 3.9.7
module load libs/cuda
source activate diffuse_vae

python apply_recon.py