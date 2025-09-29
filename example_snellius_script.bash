#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH -p gpu_mig
#SBATCH -N 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus=1
#SBATCH --output=R-%x.%j.out
#SBATCH --reservation=terv92681

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


source .venv/bin/activate # Activate virtual environment

pip install -r requirements_ass2.txt

#echo 'Starting new experiment';
python run.py