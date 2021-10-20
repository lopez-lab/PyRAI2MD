######################################################
#
# PyRAI2MD 2 module for thermostat in NVT ensemble
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import numpy as np

def NoseHoover(traj):
    """ Velocity scaling function in NVT ensemble (Nose Hoover thermostat)

        Parameters:          Type:
            traj             class       trajectory class

        Attribute:           Type:
            natom            int         number of atoms
            temp             float       temperature
            kinetic          float	 kinetic energy
            Vs               list        additional velocity information
            kb               float       Boltzmann's constant
            fs_to_au         float       unit conversion fs to au of time

    """

    natom    = traj.natom
    kinetic  = traj.kinetic
    temp     = traj.temp
    size     = traj.size
    Vs       = traj.Vs
    kb       = 3.16881 * 10**-6
    fs_to_au = 2.4188843265857 * 10**-2

    if len(Vs) == 0:
        freq = 1 / (22 / fs_to_au) ## 22 fs to au Hz
        Q1 = 3 * natom * temp * kb / freq**2
        Q2 = temp * kb / freq**2
        traj.Vs = [Q1, Q2, 0, 0]

    else:
        Q1, Q2, V1, V2 = Vs
        G2 = (Q1 * V1**2 - temp * kb) / Q2
        V2 += G2 * size / 4
        V1 *= np.exp(-V2 * size / 8)
        G1 = (2 * kinetic - 3 * natom * temp * kb) / Q1
        V1 += G1 * size / 4
        V1 *= np.exp(-V2 * size / 8)
        s = np.exp(-V1 * size / 2)

        traj.kinetic *= s**2
        traj.velo    *= s

        V1 *= np.exp(-V2 * size / 8)
        G1 = (2 * kinetic - 3 * natom * temp * kb) / Q1
        V1 += G1 * size / 4
        V1 *= np.exp(-V2 * size / 8)
        G2 = (Q1 * V1**2 - temp * kb) / Q2
        V2 += G2 * size / 4

        traj.Vs = [Q1, Q2, V1, V2]

    return traj
