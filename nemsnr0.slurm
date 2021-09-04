#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --constraint=rtx_8000
source /usr/ebuild/software/Anaconda3/2020.07/etc/profile.d/conda.sh
conda activate hc_env

echo "it begins"
nvidia-smi
python -u snr0.py > snr0.out 2>&1