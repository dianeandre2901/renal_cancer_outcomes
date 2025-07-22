#!/bin/bash
#PBS -N surformer_cox_tab
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=20:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/surformer_cox_tab.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/surformer_cox_tab.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/surformer_cox_img_tab.py

echo "Job ended at $(date)"