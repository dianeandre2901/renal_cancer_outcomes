#!/bin/bash
#PBS -N model4_60slides_aftertuning
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/model4__aftertuning.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/model4__aftertuning.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/model4_coords_extract_aftertuning.py

echo "Job ended at $(date)"