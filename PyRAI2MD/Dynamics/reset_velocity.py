######################################################
#
# PyRAI2MD 2 module for translation and rotation velocity removal
#
# Author Jingbai Li
# Sep 5 2021
#
######################################################

import numpy as np
from numpy import linalg as la

def ResetVelo(traj):
    """ Removing translation and rotation velocity

        Parameters:          Type:
            traj             class    trajectory class

        Return:              Type:
            traj             class    trajectory class

    """

    iter = traj.iter     # current MD step
    xyz  = traj.coord    # cartesian coordiante in angstrom (Nx3)
    velo = traj.velo     # velocity in Eh/Bohr (Nx3)
    GD   = traj.graddesc # gradient descent
    M    = traj.mass     # mass matrix in ams unit (Nx1)
    test = 0             # debug mode

    ## in gradient descent, do not reset velocity since they are zero
    if GD == 1:
        return traj

    ## find center of mass and momentum of inertia as principal axis
    ## then project velocity and position vector to principal axis system
    com, paxis, caxis = Inertia(xyz, M)
    pvelo = np.dot(velo, paxis)
    pxyz = np.dot(xyz - com, paxis)

    ## find the translation and augular velocity at center of mass
    vcom = GetVCOM(velo, M)
    wcom = GetWCOM(pxyz, pvelo, M)

    ## first remove the translation with un-projected velocity
    velo1 = RmVCOM(velo, vcom)
    vcom1 = GetVCOM(velo1, M)

    ## then project the new velocity to principal axis system
    ## find the angular velocity then remove it
    pvel1 = np.dot(velo1, paxis)
    wcom1 = GetWCOM(pxyz, pvel1, M)
    velo2 = RmWCOM(pxyz, pvel1, wcom1, caxis)

    ## compute kinetic energy for original, translation removed, and translation/rotation removed velocity
    K1 = 0.5 * np.sum(M * velo**2)
    K2 = 0.5 * np.sum(M * velo1**2)
    K3 = 0.5 * np.sum(M * velo2**2)

    ## scale the new velocty to conserve kinetic energy
    velo_noTR = velo2 * (K1 / K3)**0.5

    if test == 1:
        vcom2 = GetVCOM(velo2, M)
        pvel2 = np.dot(velo2, paxis)
        wcom2 = GetWCOM(pxyz, pvel2, M)

        #print('Original')
        #print('Principla axis')
        #print(paxis)
        #print('Cartesian axis')
        #print(caxis)
        print('Iter: ', iter)
        print('Original: VCOM ',vcom, 'WCOM ', wcom,  'K ', K1)
        print('Rm Trans: VCOM ',vcom1,'WCOM ', wcom1, 'K ', K2)
        print('Rm Tr\Rr: VCOM ',vcom2,'WCOM ', wcom2, 'K ', K3)
        print('E_Tr ',K1-K2,'E_Rr ',K2-K3)

    traj.velo = np.copy(velo_noTR)

    return traj

def kabsch(P, Q):
    ## This function use Kabsch algorithm to reduce RMSD by rotation

    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:                    # ensure right-hand system
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    P = np.dot(P, U)
    diff = P-Q
    N = len(P)
    return np.sqrt((diff * diff).sum() / N)

def CheckMirror(coord, I):
    coord1 = np.dot(coord, I)
    coord2 = np.dot(coord, -I)
    rmsd1 = kabsch(coord, coord1)
    rmsd2 = kabsch(coord, coord2)

    if rmsd1 <= rmsd2:
        Im = I
    else:
        Im = -I

    return Im

def Inertia(xyz, M):
    ## this function compute momentum of intertia as principal axis

    natom = len(xyz)
    com = np.sum(M * xyz, axis=0) / np.sum(M)
    body = xyz - com

    ## initialize momentum of inertia (3x3)
    I = np.zeros([3, 3])

    ## compute momentum of inertia
    for n, i in enumerate(body):
        I += M[n][0] * (np.sum(i**2) * np.diag(np.ones(3)) - np.outer(i, i))

    ## compute principal axis
    eigval, eigvec = np.linalg.eig(I)
    prin_axis = CheckMirror(xyz, eigvec)
    cart_axis = la.inv(prin_axis)

    return com, prin_axis, cart_axis

def GetCOM(xyz, M):
    ## This function compute velocity center of mass

    ccom = np.sum(M * xyz, axis=0) / np.sum(M)

    return com

def GetVCOM(velo, M):
    ## This function compute velocity at center of mass

    vcom = np.sum(M * velo, axis=0) / np.sum(M)

    return vcom

def RmVCOM(velo, vcom):
    ## This function remove velocity at center of mass from velocity on each atom

    new_velo = velo - vcom

    return new_velo

def GetWCOM(xyz, velo, M):
    ## This function compute angular velocity about momentum of inertia as principal axis
    ## xyz and velo are projected to principal axis

    ## inital average angular velocity matrix and average momentum of inertia
    wcom = np.zeros(3)
    J = np.zeros([3, 3])

    ## compute angular velocity and momentum of inertia
    for n, i in enumerate(xyz):
        w = np.cross(i, velo[n]) / np.sum(i**2)
        j = M[n][0]*(np.sum(i**2) * np.diag(np.ones(3)) - np.outer(i, i))
        wcom += np.dot(j, w)
        J += j

    wcom = np.dot(la.inv(J), wcom)

    return wcom

def RmWCOM(xyz, velo, wcom, cart_axis):
    ## This function removes angular velocity about momentum of inertia as principal axis from velocity on each atom
    ## xyz and velo are	projected to principal axis
    ## new_vele is projected back to reference axis

    new_velo = []
    for n, i in enumerate(velo):
        linear = np.cross(wcom, xyz[n])
        radial = i - linear
        new_velo.append(radial)

    new_velo = np.array(new_velo)
    new_velo = np.dot(new_velo, cart_axis)

    return new_velo
