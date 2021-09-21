#!/bin/bash
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --constraint=rtx_6000

echo "it begins"
nvidia-smi
# conda deactivate
# source /nfs/rs/sjkim/users/chenhao1/miniconda3/etc/profile.d/conda.sh
# source activate
which conda
python -u debug.py > debug.out 2>&1
echo "done"