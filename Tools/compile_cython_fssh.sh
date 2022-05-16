######################################################
#
# PyRAI2MD 2 script for compling fssh cython library
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

cp setup.py ../
cd ..
python3 setup.py build_ext --inplace
rm -r build
rm setup.py
cd Tools
echo "COMPLETE !"

