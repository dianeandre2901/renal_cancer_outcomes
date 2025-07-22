#!/bin/bash
#PBS -N optunatry
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=16:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/_optunatry.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/_optunatry.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python optunatry.py

echo "Job ended at $(date)"