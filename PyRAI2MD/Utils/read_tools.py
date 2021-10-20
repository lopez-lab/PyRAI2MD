######################################################
#
# PyRAI2MD 2 module for utility tools - read index/file
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os
import numpy as np

class ReadVal:
    """ This class read value and convert datatype
        Parameters:          Type:
            data_type        str         index type
            x                list        a list of string of values

        Return:              Type:
            x                *           str, int, float, list
    """
    def __init__(self, data_type = 's'):
        self.type = data_type

    def __call__(self, x):
        self._data_func = {
            's'    : self._string,
            'i'    : self._integer,
            'f'    : self._floatnum,
            'il'   : self._integer_list,
            'fl'   : self._floatnum_list,
            'b'    : self._boolean,
            }
        return self._data_func[self.type](x)

    def _string(self, x):
        return str(x[0])

    def _integer(self, x):
        return int(x[0])

    def _floatnum(self, x):
        return float(x[0])

    def _integer_list(self, x):
        return [int(f) for f in x]

    def _floatnum_list(self, x):
        return [float(f) for f in x]

    def _boolean(self, x):
        B = {
            'true' : True,
            '1'    : True,
            'false': False,
            '0'    : False,
            }
        return B[x[0].lower()]

class ReadIndex:
    """ This class read individual or a group of index from a list or a file
        Parameters:          Type:
            index_type       str         index type
            var              list        a list of string of index or file

        Return:              Type:
            index            list        index list
    """
    def __init__(self, index_type = 's'):
        self.type = index_type

    def __call__(self, var):
        if self.type != 'g':
            return self._read_index(var)
        else:
            return self._read_index_group(var)

    def _get_index(self, index):
        ## This function read single, range, separete range index and convert them to a list
        index_list = []
        for i in index:
            if '-' in i:
                a, b = i.split('-')
                a, b = int(a), int(b)
                index_list += range(a, b + 1)
            else:
                index_list.append(int(i))

        index_list = sorted(list(set(index_list))) # remove duplicates and sort from low to high
        return index_list

    def _read_index(self, var):
        ## This function read a group of index from a list or a file
        file = var[0]
        if os.path.exists(file) == True:
            with open(file, 'r') as indexfile:
                indices = indexfile.read().split()
        else:
            indices = var

        index_list = self._get_index(indices)
        return index_list

    def _read_index_group(self, var):
        ## This function read a group of index from a list or a file
        file = var[0]
        if os.path.exists(file) == True:
            with open(file,'r') as indexfile:
                indices = indexfile.read().splitlines()
            indices = [x.split() for x in indices]
        else:
       	    indices = ' '.join(var).split(',')
       	    indices = [x.split() for x in indices]

        index_group = [self._get_index(x) for x in indices]
        return index_group
