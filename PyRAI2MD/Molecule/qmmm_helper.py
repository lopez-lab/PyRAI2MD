######################################################
#
# PyRAI2MD 2 module for automatic qmmmm boundray detection
#
# Author Jingbai Li
# Sep 5 2021
#
######################################################

import numpy as np

from PyRAI2MD.Molecule.atom import Atom

def AutoBoundary(coord, high):
    """ Auto qmmm boudnary detection function

        Parameters:          Type:
            coord            ndarray    full nuclear coordinate
            high             ndarray    index of atoms in high level region
            pbc              ndarray    periodic boundary condition

        Return:              Type:
            boundary         ndarray    atom index at high and low level boundary

    """

    link     = []
    boundary = []

    ## under construction ...

    return link, boundary

def ComputeHcap(atoms, coord, boundary):
    """ Auto qmmm boudnary detection function

        Parameters:          Type:
            coord            ndarray    full nuclear coordinate
            boundary         ndarray    atom index at high and low level boundary
       	    pbc	       	     ndarray   	periodic boundary condition

        Return:              Type:
            Hcap             ndarray    Hcap coordinates
            Jacob            ndarray    Jacobian between caped and uncaped coordinates
    """

    ## under construction ...

    scaling = {
    'C-C'   : 0.843,
    'C-O'   : 0.854,
    'C-N'   : 0.823,
       }

    Hcap = []
    rh = Atom('H').get_radii()
    for pair in boundary:
        t, o = pair
        ao = atoms[o][0]
        at = atoms[t][0]
        ro = Atom(ao).get_radii()
        rt = Atom(at).get_radii()
        f = (ro + rh) / (ro + rt) * scaling['-'.join(sorted([ao, at]))]
        Hcap.append((coord[t] - coord[o]) * f + coord[o])
    Hcap = np.array(Hcap)
    Jacob = 1

    return Hcap, Jacob
