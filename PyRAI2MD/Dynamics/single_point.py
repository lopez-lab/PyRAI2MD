######################################################
#
# PyRAI2MD 2 module for single point calculation
#
# Author Jingbai Li
# Sep 30 2021
#
######################################################

import os, time
import numpy as np

from PyRAI2MD.Utils.timing import WhatIsTime, HowLong
from PyRAI2MD.Utils.coordinates import PrintCoord

class SinglePoint:
    """ Single point calculation class

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary
            qm               class       QM method class
            id               int         trajectory id index
            dir              boolean     create a subdirectory

        Attributes:          Type:
            version          str         version information header
            title            str         calculation title
            silent           int         silient mode for screen output
            verbose          int         verbose level of output information

        Functions:           Returns:
            run              None        run a single point calculation

    """
    def __init__(self, trajectory = None, keywords = None, qm = None, id = None, dir = None):
        self.timing = 0  ## I use this to test calculation time

        ## initialize variables
        self.version       = keywords['version']
        self.title         = keywords['control']['title']
        self.verbose       = keywords['control']['verbose']
        self.silent        = keywords['control']['silent']

        ## update calculation title if the id is available
        if id != None:
            self.title = '%s-%s' % (self.title, id)

        ## setup molecular dynamics calculation path
        if dir != None:
            self.logpath = '%s/%s' % (os.getcwd(), self.title)
            if os.path.exists(self.logpath) == False:
                os.makedirs(self.logpath)
        else:
            self.logpath = os.getcwd()

        ## create a trajectory object
        self.traj = trajectory

        ## create an electronic method object
        self.QM = qm

    def _potential_energy(self):
        self.traj = self.QM.evaluate(self.traj)

        return self.traj

    def _heading(self):
        state_info = ''.join(['%4d' % (x+1) for x in range(len(self.traj.statemult))])
        mult_info = ''.join(['%4d' % (x) for x in self.traj.statemult])

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |             Single Point Calculation              |
 |                                                   |
 *---------------------------------------------------*


 State order:      %s
 Multiplicity:     %s

 QMMM key:         %s
 QMMM xyz:         %s
 Active atoms:     %s
 Inactive atoms:   %s
 Link atoms:       %s
 Highlevel atoms:  %s
 Lowlevel atoms:   %s
 
""" % ( self.version,
        state_info,
        mult_info,
        self.traj.qmmm_key,
        self.traj.qmmm_xyz,
        self.traj.natom,
        self.traj.ninac,
        self.traj.nlink,
        self.traj.nhigh,
        self.traj.nlow)

        return headline

    def	_chkpoint(self):
        ## prepare logfile info
        log_info = '\n'.join([' State %2d: %28.16f' % (
            n + 1, x) for n, x in enumerate(self.traj.energy)]) + '\n'

        ## add verbose info
        log_info += self._verbose_log_info(self.verbose)

        ## add error info
        if  self.traj.err_energy != None and\
            self.traj.err_grad != None and\
            self.traj.err_nac != None and\
            self.traj.err_soc != None:

            log_info += """
  &error
-------------------------------------------------------
  Energy   MaxStDev:          %-10.4f
  Gradient MaxStDev:          %-10.4f
  Nac      MaxStDev:          %-10.4f
  Soc      MaxStDev:          %-10.4f
-------------------------------------------------------

""" % ( self.traj.err_energy,
        self.traj.err_grad,
        self.traj.err_nac,
        self.traj.err_soc)

        ## print log on screen
        if self.silent == 0:
            print(log_info)

        ## write logfile to disk
        self._dump_to_disk(
            self.logpath,
            self.title,
            log_info)

    def _verbose_log_info(self, verbose):
        log_info = ''

        if verbose == 0:
            return log_info

        log_info += """
  &coordinates in Angstrom
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (PrintCoord(np.concatenate((self.traj.atoms, self.traj.coord), axis = 1)))

        log_info += """
  &velocities in Bohr/au
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (PrintCoord(np.concatenate((self.traj.atoms, self.traj.velo), axis = 1)))

        for n in range(self.traj.nstate):
            try:
                grad = self.traj.grad[n]
                log_info += """
  &gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (n + 1, PrintCoord(np.concatenate((self.traj.atoms, grad), axis = 1)))

            except IndexError:
                log_info += """
  &gradient state             %3d in Eh/Bohr
-------------------------------------------------------------------------------
  Not Computed
-------------------------------------------------------------------------------
""" % (n + 1)

        for n, pair in enumerate(self.traj.nac_coupling):
            s1, s2 = pair
            m1 = self.traj.statemult[s1]
            m2 = self.traj.statemult[s2]
            try:
                coupling = self.traj.nac[n]
                log_info += """
  &nonadibatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2, PrintCoord(np.concatenate((self.traj.atoms, coupling), axis = 1)))

            except IndexError:
                log_info += """
  &nonadibatic coupling %3d - %3d in Hartree/Bohr M = %1d / %1d
-------------------------------------------------------------------------------
  Not computed
-------------------------------------------------------------------------------
""" % (s1 + 1, s2 + 1, m1, m2)

        soc_info = ''
        for n, pair in enumerate(self.traj.soc_coupling):
            s1, s2 = pair
            m1 = self.traj.statemult[s1]
            m2 = self.traj.statemult[s2]
            try:
                coupling = self.traj.soc[n]
                soc_info += '  <H>=%10.4f            %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    coupling, s1 + 1, s2 + 1, m1, m2)

            except:
                soc_info += '  Not computed              %3d - %3d in cm-1 M1 = %1d M2 = %1d\n' % (
                    s1 + 1, s2 + 1, m1, m2)

        if len(self.traj.soc_coupling) > 0:
            log_info += """
  &spin-orbit coupling
-------------------------------------------------------------------------------
%s-------------------------------------------------------------------------------
""" % (soc_info)

        return log_info

    def _dump_to_disk(self, logpath, title, log_info):
        ## output data to disk
        with open('%s/%s.log' % (logpath, title), 'a') as log:
            log.write(log_info)

    def run(self):
        start = time.time()

        ## add heading to new output files
        heading = 'Single Point Calculation Start: %20s\n%s' % (WhatIsTime(), self._heading())
        with open('%s/%s.log' % (self.logpath, self.title), 'w') as log:
            log.write(heading)

        if self.silent == 0:
            print(heading)

        ## single point
        self._potential_energy()

        ## checkpointing
        self._chkpoint()

        end = time.time()
        walltime = HowLong(start, end)
        tailing = 'Single Point Calculation End: %20s Total: %20s\n' % (WhatIsTime(), walltime)

        if self.silent == 0:
            print(tailing)

        with open('%s/%s.log' % (self.logpath, self.title),'a') as log:
            log.write(tailing)

        return self.traj
