#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -p gpu_a100
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=R-%x.%j.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 # Includes most dependencies


source .venv/bin/activate # Activate virtual environment

pip install -r requirements.txt # Install leftover dependencies

#echo 'Starting new experiment';
python 	run.py --ch_mults 1 2 2 --hidden_size 128 --lr 1e-4 --predict_frames 12 --history_frames 12 --sigma 0.12 --save --skip_test --condition_on "prior" --run_tags "cfd_eval"