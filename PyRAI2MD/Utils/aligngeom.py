######################################################
#
# PyRAI2MD 2 module for aliging molecular structures
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import numpy as np
from numpy import linalg as la
from scipy.optimize import linear_sum_assignment

def RMSD(atoms, xyz, ref):
    ## This function calculate RMSD between product and reference
    ## This function call kabsch to reduce RMSD between product and reference
    ## This function call hungarian to align product and reference

    P = xyz                     ## product coordinates
    Q = ref                     ## reference coordinates
    Patoms = [x for x in atoms]
    Qatoms = [x for x in atoms]
    P -= P.mean(axis = 0)         ## translate to the centroid
    Q -= Q.mean(axis = 0)         ## translate to the centroid

    swap = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1]])

    reflection = np.array([
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1]])

    order = []
    rmsd = []
    for s in swap:
        for r in reflection:
            Tatoms = [x for x in Qatoms]
            T = np.array([x for x in Q])
            T = T[:, s]
            T = np.dot(T, np.diag(r))
            T -= T.mean(axis = 0)
            Ip = inertia(Patoms, P)
            It = inertia(Tatoms, T)
            U1 = rotate(Ip, It)
            U2 = rotate(Ip, -It)
            T1 = np.dot(T, U1)
            T2 = np.dot(T, U2)
            order1 = hungarian(Patoms, Tatoms, P, T1)
            order2 = hungarian(Patoms, Tatoms, P, T2)
            rmsd1 = kabsch(P, T[order1])
            rmsd2 = kabsch(P, T[order2])
            order += [order1, order2]
            rmsd += [rmsd1, rmsd2]
    pick = np.argmin(rmsd)
    order = order[pick]
    rmsd = rmsd[pick]
    Q = Q[order]

    return rmsd

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

def inertia(atoms, xyz, mass):
    ## This function calculate principal axis

    xyz = np.array([i for i in xyz])   # copy the array to avoid changing it
    mass = np.array(mass).reshape(-1)
    xyz -= np.average(xyz, weights = mass, axis = 0)
    xx = 0.0
    yy = 0.0
    zz = 0.0
    xy = 0.0
    xz = 0.0
    yz = 0.0
    for n, i in enumerate(xyz):
        xx += mass[n] * (i[1]**2 + i[2]**2)
        yy += mass[n] * (i[0]**2 + i[2]**2)
        zz += mass[n] * (i[0]**2 + i[1]**2)
        xy += -mass[n] * i[0] * i[1]
        xz += -mass[n] * i[0] * i[2]
        yz += -mass[n] * i[1] * i[2]

    I = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    eigval, eigvec = np.linalg.eig(I)

    return eigvec[np.argmax(eigval)]

def rotate(p, q):
    ## This function calculate the matrix rotate p onto q

    if (p == q).all():
        return np.eye(3)
    elif (p == -q).all():
        # return a rotation of pi around the y-axis
        return np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    else:
        v = np.cross(p, q)
        s = np.linalg.norm(v)
        c = np.vdot(p, q)
        vx = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
        return np.eye(3) + vx + np.dot(vx, vx) * ((1. - c) / (s * s))

def hungarian(Patoms, Qatoms, P, Q):
    ## This function use hungarian algorithm to align P onto Q
    ## This function call linear_sum_assignment from scipy to solve hungarian problem
    ## This function call inertia to find principal axis
    ## This function call rotate to rotate P onto aligned Q

    unique_atoms = np.unique(Patoms)

    reorder = np.zeros(len(Qatoms), dtype = int)
    for atom in unique_atoms:
        Pidx = []
        Qidx = []

        for n,p in enumerate(Patoms):
            if p == atom:
                Pidx.append(n)
        for m,q in enumerate(Qatoms):
            if q == atom:
                Qidx.append(m)

        Pidx = np.array(Pidx)
        Qidx = np.array(Qidx)
        A = P[Pidx]
        B = Q[Qidx]
        AB = np.array([[la.norm(a - b) for b in B] for a in A])
        Aidx, Bidx = linear_sum_assignment(AB)
        reorder[Pidx] = Qidx[Bidx]
    return reorder

def AlignGeom(x, geom_pool, mass):
    ## This function align a geometry with all train data geometries to find most similar one 
    atoms = np.array(x)[:, 0].astype(str)
    xyz = np.array(x)[:, 1: 4].astype(float)
    similar=[RMSD(atoms, xyz, np.array(geom)[:, 1: 4].astype(float)) for geom in geom_pool]

    return np.argmin(similar), np.amin(similar)
