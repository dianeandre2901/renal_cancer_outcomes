#!/bin/bash
#PBS -N model4_cox_20_img+tab
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/model4_cox_20_img+tab.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/model4_cox_20_img+tab.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/model4_cox20_img+tab.py

echo "Job ended at $(date)"