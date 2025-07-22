#!/bin/bash
#PBS -N model2_binary
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=12:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/_model2_binary.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/_model2_binary.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/model2-binary.py

echo "Job ended at $(date)"