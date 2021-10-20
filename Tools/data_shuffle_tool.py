######################################################
#
# PyRAI2MD 2 module for shuffling training data
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

import sys, json
import numpy as np
from optparse import OptionParser

def main():

    usage = """
    PyRAI2MD training data shuffle tool

    Usage:
        python3 data_shuffle_tool.py [input_file] [options]

    """

    description = ''
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-s', dest='seed',       type=float,   nargs=1, help='random number seed', default = 1234)

    (options, args) = parser.parse_args()
    seed = options.seed

    title = sys.argv[1].split('.')[0]
    with open(sys.argv[1], 'r') as indata:
        data = json.load(indata)

    natom = data['natom']
    nstate = data['nstate']
    nnac = data['nnac']
    nsoc = data['nsoc']
    xyz = data['xyz']
    energy = data['energy']
    grad = data['grad']
    nac = data['nac']
    soc = data['soc']

    size = len(xyz)
    index = np.arange(size)
    np.random.shuffle(index)

    xyz = np.array(xyz)[index].tolist()
    energy = np.array(energy)[index].tolist()
    grad = np.array(grad)[index].tolist()
    nac = np.array(nac)[index].tolist()
    soc = np.array(soc)[index].tolist()

    newset = {
        'natom'  : natom,
        'nstate' : nstate,
        'nnac'   : nnac,
        'nsoc'   : nsoc,
        'xyz'    : xyz,
        'energy' : energy,
        'grad'   : grad,
        'nac'    : nac,
        'soc'    : soc,
        }

    with open('%s-shuffled.json' % (title),'w') as outdata:
        json.dump(newset,outdata)


if __name__ == '__main__':
    main()
