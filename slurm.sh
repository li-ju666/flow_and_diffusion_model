#! /usr/bin/bash
#SBATCH --time=08:00:00
#SBATCH -A NAISS2024-22-1358
#SBATCH --gpus-per-node=A40:1

EXE="apptainer exec --nv /mimer/NOBACKUP/groups/scalablefl/containers/torch.sif python3"

${EXE} train.py

wait
echo "Done"