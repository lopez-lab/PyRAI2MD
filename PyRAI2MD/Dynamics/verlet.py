######################################################
#
# PyRAI2MD 2 module for Velocity Verlet integrator
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import numpy as np

def VerletI(traj):
    """ Velocity Verlet function 1: (V0,G0) -> R1

        Parameters:          Type:
            traj             class       trajectory class

        Attribute:           Type:
            iter             int         current iteration
            state            int         current state
            size             int         time step size
            graddesc         int         gradient descent
            coord            ndarray     the present nuclear coordinates
       	    velo  	     ndarray   	 the present nuclear velocities
       	    grad  	     ndarray   	 the present nuclear gradients
       	    mass	     ndarray   	 nuclear mass
            freeze           ndarray     index of frozen atoms

        Return:              Type:
       	    traj	     class     	 molecule class

    """

    iter     = traj.iter
    coord    = traj.coord
    velo     = traj.velo
    grad     = traj.grad
    mass     = traj.mass
    size     = traj.size
    state    = traj.state
    graddesc = traj.graddesc
    freeze   = traj.freeze

    if iter == 1:
        return traj

    ## choose gradient of current state
    grad     = grad[state - 1]

    ## remove velocity and gradient of froze atom
    if len(freeze) > 0:
        velo[freeze] = np.array([0, 0, 0])
        grad[freeze] = np.array([0, 0, 0])

    if graddesc == 1:
        velo = np.zeros(velo.shape)

    coord += (velo * size - 0.5 * grad / mass * size**2) * 0.529177249
    traj.coord = np.copy(coord)

    return traj

def VerletII(traj):
    """ Velocity Verlet function 2: (G1,G0) -> V1

        Parameters:          Type:
            traj             class	 trajectory class

        Attribute:           Type:
            iter             int         current iteration
            state            int         current state
            size             int         time step size
            graddesc         int         gradient descent
            coord            ndarray     the present nuclear coordinates
            velo             ndarray     the present nuclear velocities
            grad             ndarray   	 the present nuclear gradients
            grad1            ndarray     the previous gradients
            mass             ndarray     nuclear mass       
            freeze           ndarray     index of frozen atom

        Return:              Type:
            traj             class	 trajectory class
    """

    ## TODO use	mol to define where to update trajectories

    iter       = traj.iter
    mass       = traj.mass
    grad       = traj.grad
    grad1      = traj.grad1
    velo       = traj.velo
    size       = traj.size
    state      = traj.state
    last_state = traj.last_state
    graddesc   = traj.graddesc
    freeze     = traj.freeze

    if iter == 1:
        return traj

    grad1  = grad1[last_state - 1]
    grad   = grad[state - 1]
    velo  -= 0.5 * (grad1 + grad) / mass * size

    if graddesc == 1:
       	velo = np.zeros(velo.shape)

    ## remove velocity of froze atom
    if len(freeze) > 0:
        velo[freeze] = np.array([0, 0, 0])

    traj.velo = np.copy(velo)

    return traj
