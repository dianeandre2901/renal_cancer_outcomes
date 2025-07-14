#!/bin/bash
#PBS -N model3_deepsurv
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/model3_deepsurv2$PBS_JOBID.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/model3_deepsurv2$PBS_JOBID.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/model3_deepsurv.py

echo "Job ended at $(date)"