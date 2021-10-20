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
    cdef int nac_pair, nac_pos

    nac_pair = sorted([state - 1, new_state - 1])
    if nac_pair in nac_coupling:
        nac_pos = nac_coupling.index(nac_pair)
        nacv = nac[nac_pos]      # pick up pre-stored non-adiabatic coupling vectors between state and new_state
    else:
        sys.exit('\n  DataNotFoundError\n  PyRAI2MD: looking for nonadibatic coupling between %s and %s' % (state, new_state))

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
    cdef np.ndarray V            = traj['velo']
    cdef np.ndarray M            = traj['mass']
    cdef np.ndarray E            = traj['energy']
    cdef float      Ekin         = traj['kinetic']
    cdef list       nac_coupling = traj['nac_coupling']
    cdef list       statemult    = traj['statemult']

    cdef np.ndarray At = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray Ht = np.diag(E).astype(complex)
    cdef np.ndarray Dt = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray B = np.zeros((nstate, nstate))
    cdef np.ndarray dB = np.zeros((nstate, nstate))
    cdef np.ndarray dAdt = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray dHdt = np.zeros((nstate, nstate), dtype = complex)
    cdef np.ndarray dDdt = np.zeros((nstate, nstate), dtype = complex)

    cdef int n, i, j, k, p, stop, hoped, nhop, event, s1, s2, frustrated
    cdef float deco, z, gsum, Asum, Amm, nacme
    cdef list pair
    cdef np.ndarray Vt, g, tau, NAC

    hoped = 0
    stop = 0
    for n, pair in enumerate(nac_coupling):
        s1, s2 = pair
        nacme = np.sum(V * N[n]) / AvoidSingularity(E[s1], E[s2], s1, s2) 
        Dt[s1, s2] = nacme
        Dt[s2, s1] = -Dt[s1, s2]

    if iter == 1:
        At[state - 1, state - 1] = 1
        Vt = V
    else:
        dHdt = (Ht - H) / substep
        dDdt = (Dt - D) / substep
        nhop = 0
        
        if verbose == 2:
            print('-------------- TEST ----------------')
            print('Iter: %s' % (iter))
            print('One step')
            print('dPdt')
            print(dPdt(A,H,D))
            print('matB')
            print(matB(A+dPdt(A,H,D)*delt*substep,H,D)*delt*substep)
            print('Integral')

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
            for p in range(nstate):
                if np.real(A[p, p]) > 1 or np.real(A[p, p]) < 0:
                    A -= dAdt  # revert A
                    B -= dB
                    ## TODO p > 1 => p = 1 ; p<0 => p=0
                    stop = 1   # stop if population exceed 1 or less than 0            
            if stop == 1:
                break

            for j in range(nstate):
                if j != state - 1:
                    g[j] += np.amax([0, B[j, state - 1] * delt / np.real(A[state - 1, state - 1])])

            z = np.random.uniform(0, 1)

            gsum = 0
            for j in range(nstate):
                if statemult[j] == statemult[state - 1]:   # check for the same spin states
                    gsum += g[j]
                    nhop = np.abs(j + 1 - state)
                    if gsum > z and nhop <= maxhop:
                        new_state = j + 1
                        nhop = np.abs(j + 1 - state)
                        event = 1
                        break

            if verbose > 2:
                print('\nSubIter: %5d' % (i+1))
                print('NAC')
                print(Dt)
                print('D')
                print(D)
                print('A')
                print(A)
                print('B')
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

        At=A

    return At, Ht, Dt, Vt, hoped, old_state, state
