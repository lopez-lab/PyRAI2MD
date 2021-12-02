######################################################
#
# PyRAI2MD 2 module for scaling energy in NVE ensemble
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

def NVE(traj):
    """ Velocity scaling function in NVE ensemble

        Parameters:          Type:
            traj             class       trajectory class

        Attribute:           Type:
            iter             int         current iteration
            energy           ndarray     potential energy in the present step
            energy1          ndarray     potential energy in one step before
            kinetic          float	 kinetic energy in the present step
            kinetic1         float	 kinetic energy in one step before
            last_state       int         the previous state
            state            int         the present state

        Return:              Type:
            traj             dict        trajectory class

    """

    iter       = traj.iter
    energy     = traj.energy
    energy1    = traj.energy1
    kinetic    = traj.kinetic
    kinetic1   = traj.kinetic1
    state      = traj.state
    last_state = traj.last_state

    if iter > 1:
        total_energy = energy1[last_state - 1] + kinetic1
        target_kinetic = total_energy - energy[state - 1]
       	if target_kinetic > 0:
            s = (target_kinetic / kinetic)**0.5
        else:
            s = 1 # do not scale negative velocity
        traj.kinetic *= s**2
        traj.velo *= s

    ## reset other thermostat
    traj.Vs = []

    return traj

