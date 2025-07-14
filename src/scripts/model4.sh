#!/bin/bash
#PBS -N model4_patchsizenone
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=18:00:00
#PBS -o /rds/general/user/dla24/home/thesis/src/results/logs/model4_patchsizenone$PBS_JOBID.log
#PBS -e /rds/general/user/dla24/home/thesis/src/results/logs/model4_patchsizenone$PBS_JOBID.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python /rds/general/user/dla24/home/thesis/src/models/model4_coords_extract.py

echo "Job ended at $(date)"