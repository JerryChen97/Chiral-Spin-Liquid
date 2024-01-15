#!/bin/bash
#SBATCH --job-name=iDMRGChiralSpinLiquid
#
#SBATCH --account=rrg-yche
#
#SBATCH --time=23:59:00
#
#SBATCH --mem=16000
#SBATCH --cpus-per-task=8
#
#SBATCH --mail-user=ychen2@perimeterinstitute.ca
#SBATCH --mail-type=all

echo 'Starting:'
python iDMRG.py
echo 'Complete.'
