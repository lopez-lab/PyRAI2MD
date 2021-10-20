######################################################
#
# PyRAI2MD 2 module for ML-QC mixed molecular dynamics
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

import time, os, pickle
import numpy as np

from PyRAI2MD.Dynamics.aimd import AIMD

class MIXAIMD(AIMD):
    """ Ab initial molecular dynamics class

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary
            qm               class       QM method class
            ref              class       reference QM method
            id               int         trajectory id index
            dir              boolean     create a subdirectory

        Attributes:          Type:
            ref_energy       int         use reference energy for hybrid ML/QM molecular dynamics
            ref_grad         int         use reference gradient for hybrid ML/QM molecular dynamics
            ref_nac          int         use reference nac for hybrid ML/QM molecular dynamics
            ref_soc          int         use reference soc for hybrid ML/QM molecular dynamics

        Functions:           Returns:
            run              class       run molecular dynamics simulation

    """

    def __init__(self, trajectory = None, keywords = None, qm = None, ref = None, id = None, dir = None):
        super().__init__(trajectory = trajectory, keywords = keywords, qm = qm, id = id, dir = dir)
        ## initialize variables for mixed dynamics
        self.ref_energy  = keywords['md']['ref_energy']
        self.ref_grad    = keywords['md']['ref_grad']
        self.ref_nac     = keywords['md']['ref_nac']
        self.ref_soc     = keywords['md']['ref_soc']

        ## create a reference electronic method object
        self.REF = ref

    def _potential_energies(self, traj):
        ## modify the potential energy calcultion to mixed mode
        traj_qm = self.QM.evaluate(traj)
        traj_ref = self.REF.evaluate(traj)
        traj_mix = self._mix_properties(traj_qm, traj_ref)

        return traj_mix

    def _mix_properties(self, traj_qm, traj_ref):
        if self.ref_energy == 1:
            traj_qm.energy = np.copy(traj_ref.energy)
        if self.ref_grad == 1:
            traj_qm.grad = np.copy(traj_ref.grad)
        if self.ref_nac == 1:
            traj_qm.grad = np.copy(traj_ref.nac)
        if self.ref_soc == 1:
            traj_qm.soc = np.copy(traj_ref.soc)

        return traj_qm
