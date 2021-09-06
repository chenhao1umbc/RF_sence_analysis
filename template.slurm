#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --constraint=rtx_6000

echo "it begins"
nvidia-smi
conda avtivate base
which conda
python -u speed_test.py > speed_test.out 2>&1