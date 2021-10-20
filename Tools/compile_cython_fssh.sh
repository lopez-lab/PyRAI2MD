######################################################
#
# PyRAI2MD 2 script for compling fssh cython library
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

python3 setup.py build_ext --inplace
rm -r build
echo "COMPLETE !"
