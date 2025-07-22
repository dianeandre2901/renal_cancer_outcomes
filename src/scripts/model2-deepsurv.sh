#!/bin/bash
#PBS -N model2_deepsurv
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=6:00:00
#PBS -o _model2_deepsurv.log
#PBS -e model2_deepsurv.err



cd $PBS_O_WORKDIR

source ~/miniforge3/bin/activate thesis-hpc

echo "Job started at $(date) on $(hostname)"

python model2_deepsurv.py

echo "Job ended at $(date)"