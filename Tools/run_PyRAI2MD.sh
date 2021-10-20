#!/bin/sh
## script for PyRAI2MD
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:59:59
#SBATCH --job-name=tod-8mf-1
#SBATCH --partition=large,long,short,lopez
#SBATCH --mem=11000mb
#SBATCH --output=%j.o.slurm
#SBATCH --error=%j.e.slurm

export INPUT=input
export WORKDIR=/scratch/lijingbai2009/R-TOD/github/ML_NAMD_demos/TOD-8Me/tod-8me-1
export PYRAI2MD=/home/lijingbai2009/share/NN-ChemI/PyRAIMD2/pyrai2md
export PATH=/work/lopez/Python-3.7.4/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/lopez/Python-3.7.4/lib

cd $WORKDIR
python3 $PYRAI2MD/pyrai2md.py $INPUT

