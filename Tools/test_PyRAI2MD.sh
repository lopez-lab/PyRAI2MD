#!/bin/sh
## script for PyRAIMD

##### ------ Setup PyRAI2MD ------
export PYRAI2MD=/home/lijingbai2009/share/NN-ChemI/PyRAIMD2/pyrai2md/ 
export PATH=/work/lopez/Python-3.7.4/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/lopez/Python-3.7.4/lib

##### ------ Setup BAGEL  ------
export BAGEL=/work/lopez/Bagel-mvapich
export MPI=/work/lopez/mvapich2-2.3.4
export BLAS=/work/lopez/BLAS
export LAPACK=/work/lopez/BLAS
export BOOST=/work/lopez/Boost/
export MKL=/work/lopez/intel/mkl/bin/mklvars.sh
export ARCH=intel64

##### ------ Setup MOLCAS ------
export MOLCAS=/work/lopez/Molcas
export TINKER=/work/lopez/Molcas/tinker-6.3.3/bin

##### ------ Run PyRAI2MD testcases ------
python3 $PYRAI2MD/pyrai2md.py quicktest

