#!/bin/bash
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=24:00:00 
#PBS -j oe
#PBS -N patch_extraction512npz

cd $PBS_O_WORKDIR

# Load conda/miniforge if needed (some clusters require a module load, ask your admins)
# module load anaconda/miniforge3

# Activate your environment properly
source /rds/general/user/dla24/home/miniforge3/bin/activate r-env

# Check python is available
which python
python --version

# Run Python script
python extract_patches512_npz.py



