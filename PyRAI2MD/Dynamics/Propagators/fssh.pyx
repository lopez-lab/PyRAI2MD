######################################################
#
# PyRAI2MD 2 module for fewest swithest surface hopping
#
# Author Jingbai Li
# Sep 7 2021
#
######################################################

import sys
import numpy as np
cimport numpy as np
from PyRAI2MD.Dynamics.Propagators.tsh_helper import AvoidSingularity, AdjustVelo

cdef kTDC(int s1, int s2, np.ndarray E, np.ndarray Ep, np.ndarray Epp, float dt):
    """ Computing the curvature-driven time-dependent coupling
    The method is based on Truhlar et al J. Chem. Theory Comput. 2022 DOI:10.1021/acs.jctc.1c01080

        Parameters:          Type:
            s1               int        state 1
            s2               int        state 2
            E                ndarray    potential energy in the present step
            Ep               ndarray    potential energy in one step before
            Epp              ndarray    potential energy in two step before
            dt               float      time step

        Return:
            nacme            float      time-dependent nonadiabatic coupling

    """

    cdef float nacme, d2Vdt2, dVt, dVt_dt, dVt_2dt

    dVt = AvoidSingularity(E[s1], E[s2], s1, s2)
    dVt_dt = AvoidSingularity(Ep[s1], Ep[s2], s1, s2)
    dVt_2dt = AvoidSingularity(Epp[s1], Epp[s2], s1, s2)
    d2Vdt2 = (dVt - 2 * dVt_dt + dVt_2dt) / dt ** 2
    if d2Vdt2 / dVt > 0:
        nacme = (d2Vdt2 / dVt) ** 0.5 / 2
    else:
        nacme = 0

    return nacme

cdef dPdt(np.ndarray A, np.ndarray H, np.ndarray D):
    """ Computing the time derivative of state density for FSSH
    The algorithm is based on Tully's method.John C. Tully, J. Chem. Phys. 93, 1061 (1990)

        Parameters:          Type:
            A                ndarray	 state density
            H                ndarray     model Hamiltonian   
            D                ndarray     nonadiabatic coupling

        Return:              Type:
            dA               ndarray	 time derivative of state density

    """
    
    cdef int nstate = len(A)
    cdef int k, j, l
    cdef np.ndarray dA = np.zeros((nstate, nstate), dtype = complex)

    for k in range(nstate):
        for j in range(nstate):
            for l in range(nstate):
                dA[k, j] += A[l, j] * (-1j * H[k, l] - D[k, l]) - A[k, l] * (-1j * H[l, j] - D[l, j])

    return dA

cdef matB(np.ndarray A, np.ndarray H, np.ndarray D):
    """ Computing the B matrix for FSSH
    The algorithm is based on Tully's method.John C. Tully, J. Chem. Phys. 93, 1061 (1990)

        Parameters:          Type:
            A                ndarray     state density
            H                ndarray     model Hamiltonian
            D                ndarray     nonadiabatic coupling

        Return:              Type:
            b                ndarray	 the B matrix

    """

    cdef int nstate = len(A)
    cdef np.ndarray b = np.zeros((nstate, nstate))

    for k in range(nstate):
        for j in range(nstate):
            b[k, j] = 2 * np.imag(np.conj(A[k, j]) * H[k, j]) -2 * np.real(np.conj(A[k, j]) * D[k, j])

    return b

cpdef GetNAC(int state, int new_state, list nac_coupling, np.ndarray nac, int natom):
    """ Pick up non-adibatic coupling vectors from pre-stored array
        Parameters:          Type:
            state            the current state
            new_state        the new state
            nac_coupling     non-adiabatic coupling pair list
            nac              non-adiabatic coupling array

        Return
            nacv             non-adibatic coupling vectors

    """
    cdef list nac_pair
    cdef int nac_pos
    cdef np.ndarray nacv

    nac_pair = sorted([state - 1, new_state - 1])
    if nac_pair in nac_coupling and len(nac) > 0:
        nac_pos = nac_coupling.index(nac_pair)
        nacv = nac[nac_pos]      # pick up pre-stored non-adiabatic coupling vectors between state and new_state
    else:
        # if the nac vector does not exsit, return an unity matrix
        nacv = np.ones((natom, 3))

    return nacv

cpdef FSSH(dict traj):
    """ Computing the fewest swichest surface hopping
    The algorithm is based on Tully's method.John C. Tully, J. Chem. Phys. 93, 1061 (1990)

        Parameters:          Type:
            traj             class       trajectory class

        Return:              Type:
            At               ndarray     the present state denesity matrix
            Ht               ndarray     the present energy matrix (model Hamiltonian)
            Dt               ndarray     the present nonadiabatic matrix
            Vt               ndarray     the adjusted velocity after surface hopping
            hoped            int         surface hopping decision
            old_state        int         the last state
            state            int         the current(new) state

    """

    cdef np.ndarray A            = traj['last_A']
    cdef np.ndarray H            = traj['last_H']
    cdef np.ndarray D            = traj['last_D']
    cdef np.ndarray N            = traj['nac']
    cdef np.ndarray S            = traj['soc']
    cdef int        substep      = traj['substep']
    cdef float      delt         = traj['delt']
    cdef int        iter         = traj['iter']
    cdef int        nstate       = traj['nstate']
    cdef int        state        = traj['state']
    cdef int        maxhop       = traj['maxh']
    cdef str        usedeco      = traj['deco']
    cdef int        adjust       = traj['adjust']
    cdef int        reflect      = traj['reflect']
    cdef int        verbose      = traj['verbose']
    cdef int        old_state    = traj['state']
    cdef int        new_state    = traj['state']
    cdef int        integrate    = traj['integrate']
    cdef str        nactype      = traj['nactype']
    cdef np.ndarray V            = traj['velo']
    cdef np.ndarray M            = traj['mass']
    cdef np.ndarray E            = traj['energy']
    cdef np.ndarray Ep           = traj['energy1']
    cdef np.ndarray Epp          = traj['energy2']
    cdef float      Ekin         = traj['kinetic']
    cdef list       nac_coupling = traj['nac_coupling']
    cdef list       soc_coupling = traj['soc_coupling']
    cdef list       statemult    = traj['statemult']

    cdef np.ndarray At = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray Ht = np.diag(E).astype(complex)
    cdef np.ndarray Dt = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray B = np.zeros((nstate, nstate))
    cdef np.ndarray dB = np.zeros((nstate, nstate))
    cdef np.ndarray dAdt = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray dHdt = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray dDdt = np.zeros((nstate, nstate), dtype = complex)

    cdef str summary, info
    cdef int n, i, j, k, p, stop, hoped, nhop, event, s1, s2, frustrated, rstate
    cdef float deco, z, gsum, Asum, Amm, nacme, revert, hop_gsum, hop_z
    cdef list pair
    cdef np.ndarray stateorder, statemap, Vt, g, tau, NAC, exceed, deplet, hop_g
    hop_g = np.zeros(0)
    hoped = 0
    stop = 0

    ## initialize nac matrix
    if iter > 2:
        for n, pair in enumerate(nac_coupling):
            s1, s2 = pair
            if nactype == 'nac':
                nacme = np.sum(V * N[n]) / AvoidSingularity(E[s1], E[s2], s1, s2) 
            elif nactype == 'ktdc':
                nacme = kTDC(s1, s2, E, Ep, Epp, delt * substep)
            Dt[s1, s2] = nacme
            Dt[s2, s1] = -Dt[s1, s2]

    ## initialize soc matrix
    for n, pair in enumerate(soc_coupling):
        s1, s2 = pair
        socme = S[n] / 219474.6  # convert cm-1 to Hartree
        Ht[s1, s2] = socme
        Ht[s2, s1] = socme

    ## initialize state index and order
    stateindex = np.argsort(E)
    stateorder = np.argsort(E).argsort()

    ## start fssh calculation
    if iter < 4:
        At[state - 1, state - 1] = 1
        Vt = V
        info = '  No surface hopping is performed'
    else:
        dHdt = (Ht - H) / substep
        dDdt = (Dt - D) / substep
        nhop = 0
        
        if verbose >= 2:
            print('-------------- TEST ----------------')
            print('Iter: %s' % (iter))
            print('Previous Population')
            print(A)
            print('Previous Hamiltonian')
            print(H)
            print('Previous NAC')
            print(D)
            print('Current Hamiltonian')
            print(Ht)
            print('Current NAC')
            print(Dt)
            print('One step population gradient')
            print('dPdt')
            print(dPdt(A,H,D))
            print('matB')
            print(matB(A+dPdt(A,H,D)*delt*substep,H,D)*delt*substep)
            print('Integration start')

        for i in range(substep):
            if integrate == 0:
                B = np.zeros((nstate, nstate))
            g = np.zeros(nstate)
            event = 0
            frustrated=0

            H += dHdt
            D += dDdt

            dAdt = dPdt(A, H, D)
            dAdt *= delt
            A += dAdt
            dB = matB(A, H, D)
            B += dB

            exceed = np.diag(np.real(A)) - 1
            deplet = 0 - np.diag(np.real(A))
            rstate = [np.argmax(exceed), np.argmax(deplet)][np.argmax([np.amax(exceed), np.amax(deplet)])]
            revert = np.amax([exceed[rstate], deplet[rstate]])
            if revert > 0:
                A -= dAdt * np.abs(revert / np.real(dAdt)[rstate, rstate])  # revert A
                B -= dB * np.abs(revert / np.real(dAdt)[rstate, rstate])
                stop = 1 # stop if population exceed 1 or less than 0

            for j in range(nstate):
                if j != state - 1:
                    g[j] += np.amax([0, B[j, state - 1] * delt / np.real(A[state - 1, state - 1])])

            z = np.random.uniform(0, 1)

            gsum = 0
            for j in range(nstate):
                gsum += g[stateindex[j]]
                nhop = np.abs(stateindex[j] - state + 1)
                if gsum > z and 0 < nhop <= maxhop:
                    new_state = stateindex[j] + 1
                    event = 1
                    hop_g = np.copy(g)
                    hop_gsum = gsum
                    hop_z = z
                    break

            if verbose > 2:
                print('\nSubIter: %5d' % (i+1))
                print('D nac matrix')
                print(D)
                print('A population matrix')
                print(A)
                print('B transition matrix')
                print(B)
                print('Probabality')
                print(' '.join(['%12.8f' % (x) for x in g]))
                print('Population')
                print(' '.join(['%12.8f' % (np.real(x)) for x in np.diag(A)]))
                print('Random: %s' % (z))
                print('old state/new state: %s / %s' % (state, new_state))

            ## detect frustrated hopping and adjust velocity
            if event == 1:
                NAC = GetNAC(state, new_state, nac_coupling, N, len(V))
                Vt, frustrated = AdjustVelo(E[state - 1], E[new_state - 1], V, M, NAC, adjust, reflect)
                if frustrated == 0:
                    state = new_state

            ## decoherance of the propagation 
            if usedeco != 'OFF':
                deco = float(usedeco)
                tau = np.zeros(nstate)

                ## matrix tau
                for k in range(nstate):
                    if k != state-1:
                        tau[k] = np.abs( 1 / AvoidSingularity(
                            np.real(H[state - 1, state - 1]), 
                            np.real(H[k, k]),
                            state - 1,
                            k)) * (1 + deco / Ekin) 

                ## update diagonal of A except for current state
                for k in range(nstate):
                    for j in range(nstate):
                        if k != state - 1 and j != state - 1:
                            A[k, j] *= np.exp(-delt / tau[k]) * np.exp(-delt / tau[j])

                ## update diagonal of A for current state
                Asum = 0.0
                for k in range(nstate):
                    if k != state - 1:
                        Asum += np.real(A[k, k])
                Amm = np.real(A[state - 1, state - 1])
                A[state - 1, state - 1] = 1 - Asum

                ## update off-diagonal of A
                for k in range(nstate):
                    for j in range(nstate):
                        if   k == state - 1 and j != state - 1:
                            A[k, j] *= np.exp(-delt / tau[j]) * (np.real(A[state - 1, state - 1]) / Amm)**0.5
                        elif k != state - 1 and j == state - 1:
                            A[k, j] *= np.exp(-delt / tau[k]) * (np.real(A[state - 1, state - 1]) / Amm)**0.5

            if stop == 1:
                break

        ## final decision on velocity
        if state == old_state:   # not hoped
            Vt = V               # revert scaled velocity
            hoped = 0
        else:
            NAC = GetNAC(state, new_state, nac_coupling, N, len(V))
            Vt, frustrated = AdjustVelo(E[old_state - 1], E[state - 1], V, M, NAC, adjust, reflect)
            if frustrated == 0:  # hoped
                hoped = 1
            else:                # frustrated hopping
                hoped = 2

        At = A

        if len(hop_g) == 0:
            hop_g = g
            hop_z = z
            hop_gsum = gsum

        summary = ''
        for n in range(nstate):
            summary += '    %-5s %-5s %-5s %12.8f\n' % (n + 1, statemult[n], stateorder[n] + 1, hop_g[n])

        info = """
    Random number:           %12.8f
    Accumulated probability: %12.8f
    state mult  level   probability 
%s
    """ % (hop_z, hop_gsum, summary)

    return At, Ht, Dt, Vt, hoped, old_state, state, info
