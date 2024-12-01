#!/bin/bash
#SBATCH --job-name=ephod
#SBATCH --account=bpms
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
##SBATCH --qos=high
#SBATCH --partition=debug
#SBATCH --output=./example.out
#SBATCH --error=./example.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gado.hpc@gmail.com

# Install conda environment
#conda env create -f ./env.yml -p ./env
#conda activate ./env



# Activate environment
cd /scratch/jgado/EpHod/
source activate /scratch/jgado/EpHod/env


# Run prediction (AAC-SVR)
python ./ephod/predict.py \
    --fasta_path "./example/sequences.fasta" \
    --save_dir ./svr_pred \
    --aac_svr 1 \
    --verbose 1 


# Run prediction (EpHod)
python ./ephod/run.py --fasta_path "./example/sequences.fasta" --save_dir "./example" --batch_size 2 --verbose 1 --save_attention_weights 1 --save_embeddings 1


