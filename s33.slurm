#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=70:00:00
#SBATCH --constraint=rtx_6000

echo "it begins"
nvidia-smi
which conda
python -u s33.py > s33.out 2>&1
echo "done"
date