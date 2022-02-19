######################################################
#
# PyRAI2MD 2 module for MOLCAS interface
#
# Author Jingbai Li
# Sep 20 2021
#
######################################################

import os, sys, shutil
import numpy as np

from PyRAI2MD.Quantum_Chemistry.qc_molcas import MOLCAS
from PyRAI2MD.Utils.coordinates import S2F, MolcasCoord

class MOLCAS_TINKER(MOLCAS):
    """ MOLCAS/TINKER QMMMM calculation interface

        Parameters:          Type:
            keywords         dict         keywords dict
            id               int          calculation index

        Attribute:           Type:
            natom            int          number of active qmmm atoms without link atoms 
            qmmm_key         str          tinker key file
            tinker           str          path to tinker

        Functions:           Returns:
            evaluate         self         run qmmm calculation
    """

    def __init__(self, keywords = None, id = None, runtype = 'qm'):
        super().__init__(keywords = keywords, id = id, runtype = runtype)
        self.qmmm_key       = keywords['molecule']['qmmm_key']
        self.tinker         = keywords['molcas']['tinker']

        ## add tinker to runscript
        self.runscript = 'export TINKER=%s\n%s' % (self.tinker, self.runscript)

    def _write_coord(self, traj):
        ## write tinker xyz file and copy tinker key

        xyz = self._update_txyz(traj)
        with open('%s/%s.xyz' % (self.calcdir, self.project), 'w') as out:
            out.write(xyz)

        shutil.copy2(self.qmmm_key, '%s/%s.key' % (self.calcdir, self.project))

    def _update_txyz(self, traj):
        ## prepare new tinker xyz file

        traj = traj.applyqmmm() # compute the H cap coordinates

        active = traj.active
        link   = traj.link
        xyz    = traj.coord
        la     = traj.Hcap_coord
        txyz   = traj.txyz

        ## check qmmm xyz info completeness
        if len(txyz) == 0:
            sys.exit('\n  VariabError\n PyRAI2MD: QMMM xyz information is missing,\
 please use tinker xyz or set the path to qmmm_xyz in &molecule or change qm to molcas in &control')

        out = '%s\n' % (txyz[0])

        for line in txyz[1:]:
            index = int(line[0]) - 1
            part1 = '%6s  %-3s' % (line[0], line[1])
            part3 = ''.join(['%6s' % x for x in line[5:]])
            if index in active:
                x, y, z = xyz[active.index(index)]
                part2 = '%12.6f%12.6f%12.6f' % (x, y, z)
            elif index in link:
                x, y, z = la[link.index(index)]
                part2 = '%12.6f%12.6f%12.6f' % (x, y, z)
            else:
                part2 = '%12s%12s%12s' % (line[2], line[3], line[4])
            out += '%s%s%s\n' % (part1, part2, part3)

        return out

    def _read_data(self, natom):
        ## read molcas logfile and pack data

        with open('%s/%s.log' % (self.calcdir, self.project), 'r') as out:
            log = out.read().splitlines()
        spin       = -1
        coord      = []
        casscf     = []
        gradient   = []
        nac        = []
        soc        = []
        soc_mtx    = []
        soc_state  = 0
        sin_state  = 0
        tri_state  = 0
        flag       = 'grad'
        for i, line in enumerate(log):
            if   """Cartesian coordinates in Angstrom""" in line:
                coord = log[i + 4: i + 4 + natom]
                coord = MolcasCoord(coord)

            elif """Final state energy(ies)""" in line:
                spin += 1
                if   """::    RASSCF root number""" in log[i+3]:
                    shift_line = 3  # normal energy output format
                    en_col = -1
                else:
                    shift_line = 5  # relativistic energy output format
                    en_col = 1
                e = [float(x.split()[en_col]) for x in log[i + shift_line: i + shift_line + self.ci[spin]]]
                casscf += e

            elif """Molecular gradients """ in line:
                flag = 'grad'

            elif """CI derivative coupling""" in line:
                flag = 'nac'

            elif """Molecular gradients, after ESPF""" in line and flag == 'grad':
                g = log[i + 8: i + 8 + natom]
                g = S2F(g)
                gradient.append(g)

            elif """Molecular gradients, after ESPF""" in line and flag == 'nac':
                n = log[i + 8: i + 8 + natom]
                n = S2F(n)
                nac.append(n)

            elif """Nr of states""" in line:
                soc_state = int(line.split()[-1])

            elif """Root nr:""" in line:
                tri_state = int(line.split()[-1])
                sin_state = soc_state - tri_state

            elif """Spin-orbit section""" in line:
                soc_dim = int(sin_state * self.mult[0] + tri_state * self.mult[1])
                soc_urt = int(soc_dim * (soc_dim + 1) / 2)
                soc_sfs = np.zeros([soc_dim, soc_dim])
                soc_mtx = np.zeros([soc_state, soc_state])

                # form soc matrix by spin free eigenstates
                for so_el in log[i+11:i+11+soc_urt]:
                    i1, s1, ms1, i2, s2, ms2, real_part, imag_part, absolute = so_el.split()
                    i1 = int(i1) - 1
                    i2 = int(i2) - 1
                    va = float(absolute)
                    soc_sfs[i1, i2] = va
                    soc_sfs[i2, i1] = va

                # reduce soc matrix intto configuration state
                for s1 in range(sin_state):
                    for s2 in range(tri_state):
                        p2 = sin_state + s2
                        first_col = int(sin_state + s2 * self.mult[1])
                        final_col = int(sin_state + (s2 + 1) * self.mult[1])
                        soc_mtx[s1, p2] = np.sum(soc_sfs[s1, first_col: final_col]**2)**0.5
                        soc_mtx[p2, s1] = soc_mtx[s1, p2]

        ## extract soc matrix elements
        if len(self.soc_coupling) > 0 and len(soc_mtx) > 0:
            for pair in self.soc_coupling:
                s1, s2 = pair
                socme = np.array(soc_mtx[s1, s2 - self.ci[0] + sin_state]).reshape(1)
                soc.append(socme)

        ## pack data
        energy   = np.array(casscf)

        if self.activestate == 1:
            gradall = np.zeros((self.nstate, natom, 3))
            gradall[self.state - 1] = np.array(gradient)
            gradient = gradall
        else:
            gradient = np.array(gradient)

        nac      = np.array(nac)
        soc      = np.array(soc)

        return coord, energy, gradient, nac, soc

    def evaluate(self, traj):
        ## main function to run Molcas calculation and communicate with other PyRAIMD modules

        ## load trajctory info
        self.ci = traj.ci
        self.mult = traj.mult
        self.soc_coupling = traj.soc_coupling
        self.nsoc = traj.nsoc
        self.nnac = traj.nnac
        self.nstate = traj.nstate
        self.state  = traj.state
        self.activestate = traj.activestate

        ## compute properties
        energy = []
        gradient = []
        nac = []
        soc = []
        completion = 0

        ## setup Molcas calculation
        self._setup_molcas(traj)

        ## run Molcas calculation
        self._run_molcas()

        ## read Molcas output files
        coord, energy, gradient, nac, soc = self._read_data(traj.natom)

        if  len(energy) >= self.nstate and\
            len(gradient) >= self.nstate and\
            len(nac) >= self.nnac and\
            len(soc) >= self.nsoc:
            completion = 1

        ## clean up
        if self.keep_tmp == 0:
            shutil.rmtree(self.calcdir)

        # update trajectory
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.copy(soc)
        traj.err_energy = None
        traj.err_grad = None
        traj.err_nac = None
        traj.err_soc = None
        traj.status = completion

        return traj
