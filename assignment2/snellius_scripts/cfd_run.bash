#!/bin/bash
#SBATCH --time=01:00:00
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
python 	run.py --ch_mults 1 2 4 --hidden_size 256 --lr 1e-4 --predict_frames 20 --history_frames 16 --sigma 0.025 --save --skip_test