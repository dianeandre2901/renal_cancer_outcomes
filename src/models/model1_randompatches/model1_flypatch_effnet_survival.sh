#!/bin/bash
#PBS -N efficientnet_survival_train
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=18:00:00
#PBS -o efficientnet_survival_train_$PBS_JOBID.log
#PBS -e efficientnet_survival_train_$PBS_JOBID.err




cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate torch-env

echo "Job started at $(date) on $(hostname)"

python model1_flypatch_effnet_survival.py

echo "Job ended at $(date)"