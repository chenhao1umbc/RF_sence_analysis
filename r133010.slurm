#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --constraint=rtx_6000

echo "it begins"
nvidia-smi
which conda
python -u rid133010.py > rid133010.out 2>&1
echo "done"