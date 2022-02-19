######################################################
#
# PyRAI2MD 2 module for hopping probability calculation
#
# Author Jingbai Li
# Set 30 2021
#
######################################################

import os, time
import numpy as np

from PyRAI2MD.Dynamics.Propagators.surface_hopping import SurfaceHopping
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong
from PyRAI2MD.Utils.coordinates import PrintCoord

class HopProb:
    """ Surface hopping probability calculation

        Parameters:          Type:
            trajectory       class       trajectory class
            keywords         dict        keyword dictionary

        Attributes:          Type:
            version          str         version information header
            title            str         calculation title
            silent           int         silient mode for screen output
            verbose          int         verbose level of output information

        Functions:           Returns:
            run              None        compute surface hopping probability

    """

    def __init__(self, trajectory = None, keywords = None, id = None, dir = None):

        ## initialize variables
        self.version       = keywords['version']
        self.title         = keywords['control']['title']
        self.datapath      = keywords['md']['datapath']
        self.silent        = keywords['md']['silent']
        self.verbose       = keywords['md']['verbose']

        ## update calculation title if the id is available
        if id != None:
            self.title = '%s-%s' % (self.title, id)

        ## setup calculation path
        if dir != None:
            self.logpath = '%s/%s' % (os.getcwd(), self.title)
            if os.path.exists(self.logpath) == False:
                os.makedirs(self.logpath)
        else:
            self.logpath = os.getcwd()

        ## create a trajectory object
        self.traj = trajectory
        self.coord = np.copy(self.traj.coord)
        self.velo = np.copy(self.traj.velo)
        self.velo1 = np.zeros(0)
        self.coord1 = np.zeros(0)
        self.coord2 = np.zeros(0)
        self.energy = np.zeros(0)
        self.energy1 = np.zeros(0)
        self.energy2 = np.zeros(0)
        self.energy3 = np.zeros(0)
        self.grad = np.zeros(0)
        self.grad1 = np.zeros(0)
        self.grad2 = np.zeros(0)
        self.kinetic = np.zeros(0)
        self.kinetic1 = np.zeros(0)
        self.kinetic2 = np.zeros(0)
        self.nac = np.zeros(0)
        self.nac1 = np.zeros(0)
        self.soc = np.zeros(0)
        self.soc1 = np.zeros(0)
        self.pop1 = np.zeros(0)
        self.seed = 0

        ## make a data record
        self.record = {
        'coord'    : 'From Input',
        'velo'     : 'From Input',
        'coord1'   : 'Not Read',
        'coord2'   : 'Not Read',
        'velo1'    : 'Not Read',
        'energy'   : 'Not Read',
        'energy1'  : 'Not Read',
        'energy2'  : 'Not Read',
        'energy3'  : 'Not Read',
        'grad'     : 'Not Read',
       	'grad1'    : 'Not Read',
       	'grad2'    : 'Not Read',
        'kinetic'  : 'Not Read',
        'kinetic1' : 'Not Read',
        'kinetic2' : 'Not Read',
        'nac'      : 'Not Read',
        'nac1'     : 'Not Read',
        'soc'      : 'Not Read',
        'soc1'     : 'Not Read',
        'pop1'     : 'Not Read',
        'seed'     : 'Not Read',
        }

    def _load_data(self):
        ## load coordinates
        coord = '%s/%s.xyz' % (self.datapath, self.title)
        coord1 = '%s.1' % (coord)
        if os.path.exists(coord1) == True:
            self.coord1 = np.loadtxt(coord1)
            self.record['coord1'] = 'Read From %s' % (coord1)

        coord2 = '%s.2' % (coord)
        if os.path.exists(coord2) == True:
            self.coord2 = np.loadtxt(coord2)
       	    self.record['coord2'] = 'Read From %s' % (coord2)

        ## load energy
        energy = '%s/%s.energy' % (self.datapath, self.title)
        if os.path.exists(energy) == True:
            self.energy = np.loadtxt(energy)
       	    self.record['energy'] = 'Read From %s' % (energy)

       	energy1	= '%s.1' % (energy)
        if os.path.exists(energy1) == True:
       	    self.energy1 = np.loadtxt(energy1)
       	    self.record['energy1'] = 'Read From %s' % (energy1)

       	energy2	= '%s.2' % (energy)
        if os.path.exists(energy2) == True:
       	    self.energy2 = np.loadtxt(energy2)
       	    self.record['energy2'] = 'Read From %s' % (energy2)

        energy3 = '%s.3' % (energy)
        if os.path.exists(energy3) == True:
            self.energy3 = np.loadtxt(energy3)
            self.record['energy3'] = 'Read From %s' % (energy3)

        ## load grad
        grad = '%s/%s.grad' % (self.datapath, self.title)
        if os.path.exists(grad) == True:
            self.grad = np.loadtxt(grad).reshape((self.traj.nstate, self.traj.natom, 3))
       	    self.record['grad'] = 'Read From %s' % (grad)

        grad1 = '%s.1' % (grad)
        if os.path.exists(grad1) == True:
            self.grad1 = np.loadtxt(grad1).reshape((self.traj.nstate, self.traj.natom, 3))
       	    self.record['grad1'] = 'Read From %s' % (grad1)

        grad2 = '%s.2' % (grad)
        if os.path.exists(grad2) == True:
            self.grad2 = np.loadtxt(grad2).reshape((self.traj.nstate, self.traj.natom, 3))
       	    self.record['grad2'] = 'Read From %s' % (grad2)

        ## load kinetic
        kinetic = '%s/%s.kinetic' % (self.datapath, self.title)
        if os.path.exists(kinetic) == True:
            self.kinetic = np.loadtxt(kinetic)
       	    self.record['kinetic'] = 'Read From %s' % (kinetic)

        kinetic1 = '%s.1' % (kinetic)
        if os.path.exists(kinetic1) == True:
            self.kinetic1 = np.loadtxt(kinetic1)
       	    self.record['kinetic1'] = 'Read From %s' %	(kinetic1)

        kinetic2 = '%s.2' % (kinetic)
        if os.path.exists(kinetic2) == True:
            self.kinetic2 = np.loadtxt(kinetic2)
       	    self.record['kinetic2'] = 'Read From %s' % (kinetic2)

        ## load velo
        velo1 = '%s/%s.velo.1' % (self.datapath, self.title)
        if os.path.exists(velo1) == True:
            self.velo1 = np.loadtxt(velo1)
       	    self.record['velo1'] = 'Read From %s' % (velo1)

        ## load nac
       	nac = '%s/%s.nac' % (self.datapath, self.title)
        if os.path.exists(nac) == True:
       	    self.nac = np.loadtxt(nac).reshape((self.traj.nnac, self.traj.natom, 3))
       	    self.record['nac'] = 'Read From %s' % (nac)

        nac1 = '%s.1' % (nac)
        if os.path.exists(nac1) == True:
            self.nac1 = np.loadtxt(nac1).reshape((self.traj.nnac, self.traj.natom, 3))
            self.record['nac1'] = 'Read From %s' % (nac1)

        ## load soc
       	soc = '%s/%s.soc' % (self.datapath, self.title)
        if os.path.exists(soc) == True:
       	    self.soc = np.loadtxt(soc).reshape((-1))
       	    self.record['soc'] = 'Read From %s' % (soc)

       	soc1 = '%s.1' % (soc)
        if os.path.exists(soc1) == True:
            self.soc1 = np.loadtxt(soc1).reshape((-1))
       	    self.record['soc1'] = 'Read From %s' % (soc1)

        ## load population 
        pop1 = '%s/%s.pop.1' % (self.datapath, self.title)
        if os.path.exists(pop1) == True:
            self.pop1 = np.loadtxt(pop1, dtype = complex)
            self.record['pop1'] = 'Read From %s' % (pop1)

        ## load seed
        seed = '%s/%s.seed' % (self.datapath, self.title)
        if os.path.exists(seed) == True:
            self.seed = np.loadtxt(seed)[0]
            self.record['seed'] = 'Read From %s' % (seed)

    def _surfacehop(self):
        ## add info of the two and one MD step before
        self.traj.velo = self.velo1
        self.traj.coord = self.coord1
        self.traj.coord1 = self.coord2
        self.traj.energy = self.energy1
        self.traj.energy1 = self.energy2
        self.traj.energy2 = self.energy3
        self.traj.grad = self.grad1
        self.traj.grad1 = self.grad2
        self.traj.kinetic = self.kinetic1
        self.traj.kinetic1 = self.kinetic2
        self.traj.nac = self.nac1
        self.traj.soc = self.soc1
        if self.traj.sfhp == 'fssh':
            self.traj.iter = 3 # this will only initialize A, H, D
            self.traj = SurfaceHopping(self.traj)

        ## update nuclear and electronic properties
        self.traj.update_nu()
        self.traj.update_el()
        if self.record['pop1'] != 'Not Read':
            self.traj.last_A = self.pop1
        ## add info of current MD step
        self.traj.velo = self.velo
        self.traj.coord = self.coord
        self.traj.energy = self.energy
        self.traj.grad = self.grad
        self.traj.kinetic = self.kinetic
        self.traj.nac = self.nac
        self.traj.soc = self.soc
        self.traj.iter = 10

        if self.seed != 0:
            np.random.seed(self.seed)
        self.traj = SurfaceHopping(self.traj)

        return self.traj

    def _heading(self):
        state_info = ''.join(['%4d' % (x+1) for x in range(len(self.traj.statemult))])
        mult_info = ''.join(['%4d' % (x) for x in self.traj.statemult])

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |       Surface Hopping Probability Calculation     |
 |                                                   |
 *---------------------------------------------------*


 State order:  %s       
 Multiplicity: %s
 
""" % (self.version, state_info, mult_info)

        return headline

    def	_chkpoint(self):
        ## prepre logfile info
        log_info = ' Trajectory properties:\n'
        for key, location in self.record.items():
            log_info += ' %-10s %s\n' % (key, location)
        ## add surface hopping information to xyz comment line
        if   self.traj.hoped == 0:
            hop_info = ' A surface hopping is not allowed\n  **\n At state: %3d\n' % (self.traj.state)
        elif self.traj.hoped == 1:
       	    hop_info = ' A surface hopping event happened\n  **\n From state: %3d to state: %3d *\n' % (
                self.traj.last_state, self.traj.state)
        elif self.traj.hoped == 2:
            hop_info = ' A surface hopping is frustrated\n  **\n At state: %3d\n' % (self.traj.state)

        ## prepare population and potential energy info
        pop = ' '.join(['%28.16f' % (x) for x in np.diag(np.real(self.traj.A))])
        pot = ' '.join(['%28.16f' % (x) for x in self.traj.energy])

        ## prepare logfile info
        log_info += '\n Iter: %8d  Ekin = %28.16f au T = %8.2f K dt = %10d CI: %3d\n Root chosen for geometry opt %3d\n' % (
            self.traj.iter, 
            self.traj.kinetic,
            self.traj.temp,
            self.traj.size,
            self.traj.nstate,
            self.traj.last_state)

        log_info += '\n Gnuplot: %s %s %28.16f\n  **\n  **\n  **\n%s\n' % (
            pop,
            pot,
            self.traj.energy[self.traj.last_state - 1],
            hop_info)

        log_info += '''
  &surface hopping information
-------------------------------------------------------
%s
-------------------------------------------------------
''' % (self.traj.shinfo)

        ## print log on screen
        if self.silent == 0:
            print(log_info)

        with open('%s/%s.log' % (self.logpath, self.title), 'a') as log:
            log.write(log_info)

    def run(self):
        warning  = ''
        start = time.time()

        heading = 'Surface Hopping Probability Calculation Start: %20s\n%s' % (WhatIsTime(), self._heading())
        with open('%s/%s.log' % (self.logpath, self.title), 'w') as log:
            log.write(heading)

        if self.silent == 0:
            print(heading)

        ## compute surface hopping probability
        self._load_data()
        self._surfacehop()   # compute new A, H, D, state
        self._chkpoint()

        end = time.time()
        walltime = HowLong(start, end)
        tailing = 'Surface Hopping Probability Calculation End: %20s Total: %20s\n' % (WhatIsTime(), walltime)

        if self.silent == 0:
            print(tailing)

        with open('%s/%s.log' % (self.logpath, self.title),'a') as log:
            log.write(tailing)

        return self.traj
