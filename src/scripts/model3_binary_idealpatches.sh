#!/bin/bash
#PBS -N model3_binaryhead30
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/model3_binaryduplicated_try2_$PBS_JOBID.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/model3_binaryduplicated_try2_$PBS_JOBID.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/model3_binary-duplicate.py

echo "Job ended at $(date)"