######################################################
#
# PyRAI2MD 2 module for utility tools - bond library
#
# Author Jingbai Li
# Sep 25 2021
#
######################################################


def BondLib(atom1,atom2):

    label='%s-%s' % (atom1,atom2)

    bond={
    'H-H'   : 0.74,
    'H-B'   : 1.19,
    'H-C'   : 1.09,
    'H-N'   : 1.02,
    'H-O'   : 0.96,
    'H-F'   : 1.03,
    'H-Cl'  : 1.28,
    'H-Br'  : 1.41,
    'C-H'   : 1.09,
    'C-C'   : 1.20,
    'C-N'   : 1.16,
    'C-O'   : 1.16,
    'C-F'   : 1.35,
    'C-Cl'  : 1.66,
    'C-Br'  : 1.70,
    'N-H'   : 1.02,
    'N-C'   : 1.16,
    'N-N'   : 1.09,
    'N-O'   : 1.14,
    'O-H'   : 0.96,
    'O-C'   : 1.16,
    'O-N'   : 1.14,
    'O-O'   : 1.21,
    'F-F'   : 1.43,
    'F-C'   : 1.35,
    'Cl-Cl' : 1.99,
    'Cl-C'  : 1.66,
    'Br-Br' : 2.29,
    'Br-C'  : 1.70,
    }

    if label in bond.keys():
        length=bond[label]
    else:
        length=1.2

    return length
