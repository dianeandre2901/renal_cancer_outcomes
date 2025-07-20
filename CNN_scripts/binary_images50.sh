#!/bin/bash
#PBS -N binary_images50
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/CNN_scripts/logs/binary_images50.log
#PBS -e /rds/general/user/dla24/home/thesis/CNN_scripts/logs/binary_images50.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/CNN/binary_images50.py

echo "Job ended at $(date)"