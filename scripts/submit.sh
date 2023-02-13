#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=4
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=40G
# Maximum runtime of 24 hours
#SBATCH --time=48:00:00
# Name of this job
#SBATCH --job-name=SD
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=./%j.out
# stop 60 seconds early to save output
#SBATCH --signal=B:SIGINT@60

# Allow for the use of conda activate
source ~/.bashrc

# Move to submission directory
cd ${SLURM_SUBMIT_DIR}

# your job execution follows:
conda activate stable-diffusion
cd /users/j/s/jsdean/scratch/side-projects/art/stable-diffusion/

time python scripts/music_video.py
