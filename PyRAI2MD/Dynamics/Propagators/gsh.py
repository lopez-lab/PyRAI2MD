######################################################
#
# PyRAI2MD 2 module for global (generalized) surface hopping
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import sys
import numpy as np

from PyRAI2MD.Dynamics.Propagators.tsh_helper import AdjustVelo

def GSH(traj):
    """ Computing the fewest swichest surface hopping
        The algorithm is based on Zhu-Nakamura Theory, C. Zhu, Phys. Chem. Chem. Phys., 2014, 16, 25883--25895

        Parameters:          Type:
            traj             class       trajectory class

        Return:              Type:
            At               ndarray     the present state denesity matrix
            Ht               ndarray     the present energy matrix (model Hamiltonian)
            Dt               ndarray     the present nonadiabatic matrix
            Vt               ndarray     the adjusted velocity after surface hopping
            hoped            int         surface hopping decision
            old_state        int         the last state
            state            int         the new state

    """

    iter         = traj.iter
    nstate       = traj.nstate
    state        = traj.state
    verbose      = traj.verbose
    V            = traj.velo
    M            = traj.mass
    E            = traj.energy
    statemult    = traj.statemult
    maxhop       = traj.maxh
    adjust       = traj.adjust
    reflect      = traj.reflect

    # random number
    z = np.random.uniform(0, 1)

    # initialize return values
    old_state = state
    new_state = state
    ic_hop = 0
    is_hop = 0
    hoped = 0
    Vt = V
    hop_type = 'no hopping'

    # initialize state index and order
    stateindex = np.argsort(E)
    stateorder = np.argsort(E).argsort()

    # compute surface hopping probability
    if iter > 2:

        # array of approximate NAC matrix for the same spin multiplicity, unity for different spin
        N = np.ones([nstate, V.shape[0], V.shape[1]])

        # array of hopping probability
        g = np.zeros(nstate)

        # accumulated probability
        gsum = 0

        target_spin = statemult[state - 1]

        for i in range(nstate):

            # skip the present state
            if i == state - 1:
                continue

            state_spin = statemult[i]

            if state_spin == target_spin:
                P, N[i] = InternalConversionProbability(i, traj)
            else:
                P = IntersystemCrossingProbability(i, traj)

            g[i] += P

        event = 0
        for j in range(nstate):
            gsum += g[stateindex[j]]
            nhop = np.abs(stateindex[j] - state + 1)
            if gsum > z and 0 < nhop <= maxhop:
                new_state = stateindex[j] + 1
                event = 1
                break

        # if surface hopping event has occured
        if event == 1:
            # Velocity must be adjusted because hop has occurred
            Vt, frustrated = AdjustVelo(E[old_state - 1], E[state - 1], V, M, N[state - 1], adjust, reflect)

            # if hop is frustrated, revert the current state to old state
            if frustrated == 1:
                state = old_state
                hoped = 2
            else:
                state = new_state
                hoped = 1

        summary = ''
        for n in range(nstate):
            summary += '    %-5s %-5s %-5s %12.8f\n' % (n + 1, statemult[n], stateorder[n] + 1, g[n])

        info = """
    Random number:           %12.8f
    Accumulated probability: %12.8f
    state mult  level   probability
%s
    """ % (z, gsum, summary)

    else:
        info = '  No surface hopping is performed'

    # allocate zeros vector for population state density
    At = np.zeros([nstate, nstate])

    # assign state density at current state to 1
    At[new_state - 1, new_state - 1] = 1

    # Current energy matrix
    Ht = np.diag(E)

    # Current non-adiabatic matrix
    Dt = np.zeros([nstate, nstate])

    if iter > 2 and verbose >= 2:
        print(info)

    return At, Ht, Dt, Vt, hoped, old_state, new_state, info

def InternalConversionProbability(i, traj):
    """ Computing the probability of internal convertion
        The algorithm is based on Zhu-Nakamura Theory, C. Zhu, Phys. Chem. Chem. Phys., 2014, 16, 25883--25895

        Parameters:          Type:
            i                int         computing state
            traj             class       trajectory class

        Return:              Type:
            P                float       surface hopping probability
            N                ndarray     approximate non-adiabatic coupling vectors

    """

    state        = traj.state
    V            = traj.velo
    M            = traj.mass
    E            = traj.energy
    Ep           = traj.energy1
    Epp          = traj.energy2
    G            = traj.grad
    Gp           = traj.grad1
    Gpp          = traj.grad2
    R            = traj.coord
    Rp           = traj.coord1
    Rpp          = traj.coord2
    Ekinp        = traj.kinetic1
    gap          = traj.gap
    test         = 0

    # determine the energy gap by taking absolute value
    delE = np.abs([E[i] - E[state - 1], Ep[i] - Ep[state - 1], Epp[i] - Epp[state - 1]])

    # total energy in the system at time t2 (t)
    Etotp = Ep[state - 1] + Ekinp

    # average energy in the system over time period
    Ex = (Ep[i] + Ep[state - 1]) / 2

    # early stop if it does not satisfy surface hopping condition
    if np.argmin(delE) != 1 or delE[1] > gap/27.211396132 or Etotp - Ex < 0:
        P = 0
        NAC = np.zeros(V.shape)
        return P, NAC

    dE = delE[1]
    # Implementation of EQ 7
    begin_term = (-1 / (R - Rpp))
    if test == 1: print('IC  EQ 7 R & Rpp: %s %s' % (R, Rpp))
    if test == 1: print('IC  EQ 7 begin term: %s' % (begin_term))
    arg_min = np.argmin([i, state - 1])
    arg_max = np.argmax([i, state - 1])
    if test == 1: print('IC  EQ 7 arg_max/min: %s %s' % (arg_max,arg_min))

    f1_grad_manip_1 = (G[arg_min]) * (Rp - Rpp)
    f1_grad_manip_2 = (Gpp[arg_max]) * (Rp - R)
    if test == 1: print('IC  EQ 7 f1_1/f1_2: %s %s' % (f1_grad_manip_1,f1_grad_manip_1))

    F_ia_1 = begin_term * (f1_grad_manip_1 - f1_grad_manip_2)
    if test == 1: print('IC  EQ 7 done, F_1a_1: %s' % (F_ia_1))

    # Implementation of EQ 8
    f2_grad_manip_1 = (G[arg_max]) * (Rp - Rpp)
    f2_grad_manip_2 = (Gpp[arg_min]) * (Rp - R)
    F_ia_2 = begin_term * (f2_grad_manip_1 - f2_grad_manip_2)
    if test == 1: print('IC  EQ 8 done, F_1a_2: %s' % (F_ia_2))

    # approximate nonadiabatic (vibronic) couplings, which are
    # left out in BO approximation
    NAC = (F_ia_2 - F_ia_1) / (M**0.5)
    NAC = NAC / (np.sum(NAC**2)**0.5)
    if test == 1: print('IC  Approximate NAC done: %s' % (NAC))

    # EQ 4, EQ 5
    # F_A = ((F_ia_2 - F_ia_1) / mu)**0.5
    F_A = np.sum((F_ia_2 - F_ia_1)**2 / M)**0.5
    if test == 1: print('IC  EQ 4 done, F_A: %s' % (F_A))

    # F_B = (abs(F_ia_2 * F_ia_1) / mu**0.5)
    F_B = np.abs(np.sum((F_ia_2 * F_ia_1) / M))**0.5
    if test == 1: print('IC  EQ 5 done, F_B: %s' % (F_B))

    # compute a**2 and b**2 from EQ 1 and EQ 2
    # ---- note: dE = 2Vx AND h_bar**2 = 1 in Hartree atomic unit
    a_squared = (F_A * F_B) / (2 * dE**3)
    b_squared = (Etotp - Ex) * (F_A / (F_B * dE))
    if test == 1: print('IC  EQ 1 & 2 done, a^2, b^2: %s %s' % (a_squared,b_squared))

    # GOAL: determine sign in denominator of improved Landau Zener formula for switching
    # probability valid up to the nonadiabtic transition region
    #F_1 = E[i] - Epp[state - 1] # approximate slopes
    #F_2 = E[state - 1] - Epp[i] # here

    #if (F_1 == F_2):
    #    sign = 1
    #else:
    #    # we know the sign of the slope will be negative if either F_1 or
    #    # F_2 is negative but not the other positive if both positive or both negative
    #    sign = np.sign(F_1 * F_2)
    sign = np.sign(np.sum(F_ia_1 * F_ia_2))
    if test == 1: print('IC  Compute F sign done: %s' % (sign))

    # sign of slope determines computation of surface
    # hopping probability P (eq 3)
    pi_over_four_term = -(np.pi/ (4 *(a_squared)**0.5))
    if test == 1: print('IC  P numerator done: %s' % (pi_over_four_term))
    b_in_denom_term = (2 / (b_squared + (np.abs(b_squared**2 + sign))**0.5))
    if test == 1: print('IC  P denomerator done: %s' % (b_in_denom_term))
    P = np.exp(pi_over_four_term * b_in_denom_term**0.5)
    if test == 1: print('IC  P done: %s' % (P))

    return P, NAC

def IntersystemCrossingProbability(i, traj):
    """ Computing the probability of intersystem crossing
        The algorithm is based on Zhu-Nakamura Theory, C. Zhu, Phys. Chem. Chem. Phys., 2020,22, 11440-11451
        The equations are adapted from C. Zhu, Phys. Chem. Chem. Phys., 2014, 16, 25883--25895

        Parameters:          Type:
            i                int         computing state
            traj             class       trajectory class

        Return:              Type:
            P                float       surface hopping probability

    """

    state        = traj.state
    soc_coupling = traj.soc_coupling
    soc          = traj.last_soc
    M            = traj.mass
    E            = traj.energy
    Ep           = traj.energy1
    Epp          = traj.energy2
    Gp           = traj.grad1
    Ekinp        = traj.kinetic1
    gap          = traj.gapsoc
    test         = 0

    # determine the energy gap and type of crossing
    delE = [E[i] - E[state - 1], Ep[i] - Ep[state - 1], Epp[i] - Epp[state - 1]]
    #parallel = np.sign(delE[0]* delE[2])
    parallel = -1 # assume non-parallel PESs

    # total energy in the system at time t2 (t)
    Etotp = Ep[state - 1] + Ekinp

    # set hopping point energy to target state
    Ex = Ep[i]

    # early stop if it does not satisfy surface hopping condition
    if np.argmin(np.abs(delE)) != 1 or np.abs(delE[1]) > gap/27.211396132 or Etotp - Ex < 0:
        P = 0
        return P

    # early stop if it soc was not computed (ignored)
    soc_pair = sorted([state - 1, i])
    if soc_pair not in soc_coupling:
        P = 0
        return P

    # get soc coupling
    soc_pos = soc_coupling.index(soc_pair)
    if len(soc) >= soc_pos + 1:
        soclength = soc[soc_pos]
    else:
        sys.exit('\n  DataNotFoundError\n  PyRAI2MD: looking for spin-orbit coupling between %s and %s' % (state, i + 1))

    V12x2 = 2 * soclength / 219474.6  # convert cm-1 to hartree

    # Implementation of EQ 7
    F_ia_1 = Gp[state - 1]
    F_ia_2 = Gp[i]
    if test == 1: print('ISC EQ 7 done: %s' % (F_ia_1))
    if test == 1: print('ISC EQ 8 done: %s' % (F_ia_2))

    # EQ 4, EQ 5
    F_A = np.sum((F_ia_2 - F_ia_1)**2 / M)**0.5
    if test == 1: print('ISC EQ 4 done, F_A: %s' % (F_A))

    F_B = np.abs(np.sum((F_ia_2 * F_ia_1) / M))**0.5
    if test == 1: print('ISC EQ 5 done, F_B: %s' % (F_B))

    # compute a**2 and b**2 from EQ 1 and EQ 2
    # ---- note: V12x2 = 2 * SOC AND h_bar**2 = 1 in Hartree atomic unit
    a_squared = (F_A * F_B) / (2 * V12x2**3)
    b_squared = (Etotp - Ex) * (F_A / (F_B * V12x2))
    if test == 1: print('ISC EQ 1 & 2 done: %s %s' % (a_squared,b_squared))

    # GOAL: determine sign in denominator of improved Landau Zener formula for switching
    # probability at corssing region
    sign = np.sign(np.sum(F_ia_1 * F_ia_2))
    if test == 1: print('ISC Compute F sign done: %s' % (sign))

    # hopping probability P (eq 3)
    pi_over_four_term = -(np.pi/ (4 *(a_squared)**0.5))
    if test == 1: print('LZ-P numerator done: %s' % (pi_over_four_term))
    b_in_denom_term = (2 / (b_squared + (np.abs(b_squared**2 + sign))**0.5))
    if test == 1: print('LZ-P denomerator done: %s' % (b_in_denom_term))
    P = np.exp(pi_over_four_term * b_in_denom_term**0.5)
    if test == 1: print('LZ-P done: %s' % (P))
    if test == 1: print("""parallel crossing: %s
 1 - P / (P + 1) = %s
 1 - P           = %s
""" % (parallel, 1 - P / (P + 1), 1 - P))

    if parallel == 1:
        P = 1 - P / (P + 1)
    else:
        P = 1 - P

    return P

