#!/bin/bash
#PBS -N autoencoder1
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=18:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/autoencoder1$PBS_JOBID.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/autoencoder1$PBS_JOBID.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/autoencoder1.py

echo "Job ended at $(date)"