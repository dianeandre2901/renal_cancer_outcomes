#!/bin/bash
#PBS -N optunatrymodel4
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/optunamodel4.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/optunamodel4.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/optuna_model4.py

echo "Job ended at $(date)"