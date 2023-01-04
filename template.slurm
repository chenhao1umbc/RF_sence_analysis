#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=70:00:00
#SBATCH --nodelist=g09
#SBATCH --constraint=rtx_6000

echo "it begins"
nvidia-smi
which conda
python -u debug.py > debug.out 2>&1
echo "done"
date