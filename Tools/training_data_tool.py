######################################################
#
# PyRAI2MD 2 module for packing training data
#
# Author Jingbai Li
# Oct 11 2021
#
######################################################

import os, sys, json, multiprocessing
import numpy as np
from multiprocessing import Pool
from optparse import OptionParser

def main():

    usage = """
    PyRAI2MD training data tool

    Usage:
        python3 training_data_tool.py [options]

    """

    description = ''
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-i', dest='input',       type=str,   nargs=1, help='input file name.', default = 'input')
    parser.add_option('-n', dest='ncpu',        type=int,   nargs=1, help='number of cpus.', default = 1)
    parser.add_option('-p', dest='pyrai2mddir', type=str,   nargs=1, help='python to PyRAI2MD', default = None)

    (options, args) = parser.parse_args()
    input = options.input
    ncpu = options.ncpu
    pyrai2mddir = options.pyrai2mddir

    if pyrai2mddir == None:
        if 'PYRAI2MD' not in os.environ.keys():
            sys.exit('\n  VariableError\n  PyRAI2MD: environment variable PYRAI2MD is not set')

        else:
            pyrai2mddir = os.environ['PYRAI2MD']

    if os.path.exists(pyrai2mddir) == False:
        sys.exit('\n  FileNotFoundError\n PyRAI2MD: looking for PyRAI2MD in %s' % (pyrai2mddir))

    if os.path.exists(input) == False:
        sys.exit('\n  FileNotFoundError\n PyRAI2MD: looking for input file %s' % (input))

    sys.path.append(pyrai2mddir)
    from PyRAI2MD.variables import ReadInput
    from PyRAI2MD.Quantum_Chemistry.qc_molcas import MOLCAS
    from PyRAI2MD.Quantum_Chemistry.qc_bagel import BAGEL
    from PyRAI2MD.Quantum_Chemistry.qc_molcas_tinker import MOLCAS_TINKER

    method = {
        'molcas' : MOLCAS,
        'mlctkr' : MOLCAS_TINKER,
        'bagel'  : BAGEL,
        }

    with open(input) as infile:
        input_dict = infile.read().split('&')

    keywords = ReadInput(input_dict)

    file = keywords['file']['file']

    if os.path.exists(file) == False:
        sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for list file %s' % (file))

    with open(file, 'r') as infile:
        file_list = infile.read().splitlines()

    key = PrepKey(keywords)
    natom = key['natom']
    nstate = key['nstate']
    nnac = key['nnac']
    nsoc = key['nsoc']

    qm = keywords['control']['qm']
    wrapper = [[n, method[qm], f, key] for n, f in enumerate(file_list)]
    nfile = len(wrapper)
    ncpu = np.amin([nfile, ncpu])
    pool = multiprocessing.Pool(processes = ncpu)

    xyz_list = [[] for x in range(nfile)]
    energy_list = [[] for x in range(nfile)]
    grad_list = [[] for x in range(nfile)]
    nac_list = [[] for x in range(nfile)]
    soc_list = [[] for x in range(nfile)]

    n = 0
    for val in pool.imap_unordered(ReadData, wrapper):
        n += 1
        id, xyz, energy, grad, nac, soc = val
        xyz_list[id] = xyz
        energy_list[id] = energy        
        grad_list[id] = grad
        nac_list[id] = nac
        soc_list[id] = soc

        sys.stdout.write('CPU: %3d Extracting %6d/%-6d\r' % (ncpu, n, nfile))

    pool.close()

    dataset = {
        'natom'  : natom,
        'nstate' : nstate,
        'nnac'   : nnac,
        'nsoc'   : nsoc,
        'xyz'    : xyz_list,
        'energy' : energy_list,
        'grad'   : grad_list,
        'nac'    : nac_list,
        'soc'    : soc_list,
        }

    print ('\n    --- Summary ---')
    print ('natom:  %5d' % (natom))
    print ('nstate: %5d' % (nstate))
    print ('nnac:   %5d' % (nnac))
    print ('nsoc:   %5d' % (nsoc))
    print ('    --- Data shape ---')
    print ('xyz:   %30s' % (str(np.array(xyz_list).shape)))
    print ('energy:%30s' % (str(np.array(energy_list).shape)))
    print ('grad:  %30s' % (str(np.array(grad_list).shape)))
    print ('nac:   %30s' % (str(np.array(nac_list).shape)))
    print ('soc:   %30s' % (str(np.array(soc_list).shape)))

    with open('data.json','w') as outdata:
        json.dump(dataset, outdata)

def PrepKey(key):
    qm = key['control']['qm']
    natom = key['file']['natom']    
    ci = key['molecule']['ci']
    nstate = int(np.sum(ci))
    spin = key['molecule']['spin']
    coupling = key['molecule']['coupling']

    mult = []
    statemult = []
    for n, s in enumerate(ci):
        mt = int(spin[n] * 2 + 1)
        mult.append(mt)
        for m in range(s):
            statemult.append(mt)

    nac_coupling = []
    soc_coupling = []
    for n, pair in enumerate(coupling):
        s1, s2 = pair
        s1 -= 1
        s2 -= 1
        if statemult[s1] != statemult[s2]:
            soc_coupling.append(sorted([s1, s2]))
        else:
            nac_coupling.append(sorted([s1, s2]))

    nnac = len(nac_coupling)
    nsoc = len(soc_coupling)

    keywords = {
        'qm'           : qm,
        'natom'        : natom,
        'ci'           : ci,
        'nstate'       : nstate,
        'mult'         : mult,
        'statemult'    : statemult,
        'nac_coupling' : nac_coupling,
        'soc_coupling' : soc_coupling,
        'nnac'         : nnac,
        'nsoc'         : nsoc,
        'key'          : key,
        }

    return keywords

def ReadData(var):
    id, qm, f, keywords = var

    ci = keywords['ci']
    natom = keywords['natom']
    nstate = keywords['nstate']
    mult = keywords['mult']
    nac_coupling = keywords['nac_coupling']
    soc_coupling = keywords['soc_coupling']
    nnac = keywords['nnac']
    nsoc = keywords['nsoc']
    key = keywords['key']

    data = qm(keywords = key, id = 'Read')
    data.project = f.split('/')[-1]
    data.workdir = f
    data.calcdir = f
    data.ci = ci
    data.nstate = nstate
    data.mult = mult
    data.nac_coupling = nac_coupling
    data.soc_coupling = soc_coupling
    data.nnac = nnac
    data.nsoc = nsoc
    xyz, energy, grad, nac, soc = data._read_data(natom)

    return id, xyz, energy.tolist(), grad.tolist(), nac.tolist(), soc.tolist()

if __name__ == '__main__':
    main()

