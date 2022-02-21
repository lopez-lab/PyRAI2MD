######################################################
#
# PyRAI2MD 2 module for utility tools - coordinates formating
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os, sys
import numpy as np

def MolcasCoord(M):
    ## This function convert Molcas coordintes to list

    coord = []
    for line in M:
        index, a, x, y, z = line.split()[0:5]
        coord.append([a.split(index)[0], float(x), float(y), float(z)])

    return coord

def S2F(M):
    ## This function convert 1D string (e,x,y,z) list to 2D float array

    M = [[float(x) for x in row.split()[1: 4]] for row in M]
    return M

def C2S(M):
    ## This function convert 2D complex array to 2D string array

    M = [[str(x) for x in row] for row in M]
    return M

def S2C(M):
    ## This function convert 2D string array back to 2D complex array

    M = [[complex(x) for x in row] for row in M]
    return M

def VerifyXYZ(mol):
    ## This function determine the coordinate file type

    if isinstance(mol, str):

        if os.path.exists('%s.xyz' % (mol)) == True:
            with open('%s.xyz' % (mol), 'r') as input:
                xyzfile = input.read().splitlines()
            flag = xyzfile[2].split()[1]

            try:
                float(flag)
                xyztype = 'xyz'

            except ValueError:
       	       	xyztype	= 'tinker'

        else:
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for coordinate file %s' % (mol))

    elif isinstance(mol, np.ndarray):
       	xyztype	= 'array'

    elif isinstance(mol, list):
        xyztype = 'array'

    elif isinstance(mol, dict):
        xyztype = 'dict'
    else:
        xyztype = 'unknown'

    return xyztype

def ReadCoord(mol):
    ## This function read xyz and velo from files
    with open('%s.xyz'% (mol)) as xyzfile:
        file = xyzfile.read().splitlines()

    natom = int(file[0])
    atoms = []
    coord = []
    for i, line in enumerate(file[2: 2 + natom]):
        e, x, y, z = line.split()[0:4]
        atoms.append(e)
        coord.append([x, y, z])

    if   os.path.exists('%s.velo' % (mol)) == True:
        velo = ReadFloatText('%s.velo' % (mol))

    elif os.path.exists('%s.velocity.xyz' % (mol)) == True:
        velo = ReadFloatText('%s.velocity.xyz' % (mol))

    else:
        velo = np.zeros([natom, 3])

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord).astype(float)

    return atoms, coord, velo

def ReadInitcond(mol):
    ## This function read xyz and velo from inticondition list 
    natom = len(mol)
    atoms = []
    coord = []
    velo = np.zeros((natom, 3))
    for i, line in enumerate(mol):

        if len(line) >= 9:
            e, x, y, z, vx, vy, vz, m, chrg = line[0:9]
            atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = float(vx), float(vy), float(vz)

        else:
            e, x, y, z = line[0:4]
       	    atoms.append(e)
            coord.append([x, y, z])
            velo[i, 0: 3] = 0.0, 0.0, 0.0

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord).astype(float)

    return atoms, coord, velo

def PrintCoord(xyz):
    ## This function convert a numpy array of coordinates to a formatted string

    coord=''
    for line in xyz:
        e, x, y, z = line
        coord += '%-5s%24.16f%24.16f%24.16f\n' % (e, float(x), float(y), float(z))

    return coord

def Markatom(xyz, marks, prog):
    ## This function marks atoms for different basis set specification of Molcas

    new_xyz=[]

    for n,line in enumerate(xyz):
        e, x, y, z = line
        e = marks[n].split()[0]
        new_xyz.append([e, x, y, z])

    return new_xyz

def ReadTinkerKey(xyz, key, dtype):
    ## This function read tinker key and txyz file

    ## read key
    if os.path.exists(key) == False:
        sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for qmmm key file %s' % (key))

    with open(key, 'r') as keyfile:
        key = keyfile.read().splitlines()

    ## read xyz and velo
    if dtype == 'file':
        if os.path.exists(xyz) == False:
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for qmmm xyz file %s' % (xyz))

        with open(xyz, 'r') as txyzfile:
            txyz = txyzfile.read().splitlines()

        title = xyz.split('.xyz')[0]
        if   os.path.exists('%s.velo' % (title)) == True:
            velo = ReadFloatText('%s.velo' % (title))

        elif os.path.exists('%s.velocity.xyz' % (title)) == True:
            velo = ReadFloatText('%s.velocity.xyz' % (title))

        else:
            velo = np.zeros(0)

    elif dtype == 'dict':
        txyz = xyz['txyz']
        velo = xyz['velo']
        velo = np.array([x.replace('D', 'e').split() for x in velo]).astype(float)

    ## check key
    highlevel = []
    active = []
    nola = []
    qm = []
    mm = []
    la = []
    nact = -1
    atomtype = {}
    for line in key[1:]:
        line = line.split()
        if len(line) > 1:
            atype = line[0].upper()

            if atype == 'QM':
                nact += 1
                highlevel.append(nact)
                nola.append(nact)
                active.append(int(line[1]) - 1)
                atomtype[int(line[1]) - 1] = nact

            elif atype== 'MM':
                nact += 1
                nola.append(nact)
                active.append(int(line[1]) - 1)
                atomtype[int(line[1]) - 1] = nact

            elif atype == 'LA':
                la.append(int(line[1]) - 1)

    ## check velocity
    if len(velo) == 0:
        velo = np.zeros([len(nola), 3])
    else:
        velo = velo[nola]

    ## read txyz
    info = [txyz[0]]
    atoms = []
    coord = []
    inactive = []
    boundary = []
    for line in txyz[1:]:
        if len(line) > 0:
            line = line.split()
            info.append(line)
            index = int(line[0]) - 1
            if index in active:
                atoms.append(LookupAmber(line[1]))
                coord.append([float(line[2]), float(line[3]), float(line[4])])

            elif index in la:
                o = int(line[6]) - 1
                t = int(line[7]) - 1
                boundary.append([atomtype[o], atomtype[t]])

            else:
                inactive.append(index - 1)

    atoms = np.array(atoms).reshape((-1, 1))
    coord = np.array(coord)

    mol_info = {
        'atoms'     : atoms,
        'coord'     : coord,
        'velo'      : velo,
        'inact'     : inactive,
        'active'    : active,
        'link'      : la,
        'boundary'  : boundary,
        'highlevel' : highlevel,
        'txyz'      : info,
        }

    return mol_info

def ReadFloatText(txt):
    ## This function read float from text
    with open(txt, 'r') as input:
        ftxt = input.read().splitlines()

    flist = []
    for fx in ftxt:
        fx = fx.replace('D', 'e')
        flist.append(fx.split())

    farray = np.array(flist).astype(float)

    return farray

def LookupAmber(name):
    ## This function find the atom number for amber atom type

    Amber_dict = {
        'C'   : 'C',
        'CT'  : 'C',
        'CA'  : 'C',
        'CM'  : 'C',
        'CC'  : 'C',
        'CV'  : 'C',
        'CW'  : 'C',
        'CR'  : 'C',
        'CB'  : 'C',
        'C*'  : 'C',
        'CN'  : 'C',
        'CK'  : 'C',
        'CQ'  : 'C',
        'C2R' : 'C',
        'C3R' : 'C',
        'N'   : 'N',
        'NA'  : 'N',
        'NB'  : 'N',
        'NC'  : 'N',
        'N*'  : 'N',
        'N2'  : 'N',
        'N3'  : 'N',
        'O'   : 'O',
        'OW'  : 'O',
        'OH'  : 'O',
        'OS'  : 'O',
        'OT'  : 'O',
        'O2'  : 'O',
        'S'   : 'S',
        'SH'  : 'S',
        'P'   : 'P',
        'H'   : 'H',
        'HW'  : 'H',
        'HO'  : 'H',
        'HR'  : 'H',
        'HS'  : 'H',
        'HT'  : 'H',
        'HA'  : 'H',
        'HC'  : 'H',
        'H1'  : 'H',
        'H2'  : 'H',
        'H3'  : 'H',
        'H4'  : 'H',
        'H5'  : 'H',
        'HP'  : 'H',
        'LAH' : 'H',
        'Cl-' : 'Cl',
        'F'   : 'F'
        }

    if name in Amber_dict.keys():
        atom = Amber_dict[name]
    else:
        atom = name[0]
        print('Do not find atom type %s, use %s instead' % (name, atom))

    return atom
