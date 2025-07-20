#!/bin/bash
#PBS -N cox_images+tab50
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/CNN_scripts/logs/cox_images+tab50.log
#PBS -e /rds/general/user/dla24/home/thesis/CNN_scripts/logs/cox_images+tab50.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/CNN/cox_images+tab50.py

echo "Job ended at $(date)"