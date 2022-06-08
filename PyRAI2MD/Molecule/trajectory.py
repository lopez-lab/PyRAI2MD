######################################################
#
# PyRAI2MD 2 module for creating trajectory objects
#
# Author Jingbai Li
# Sep 6 2021
#
######################################################

import numpy as np
from PyRAI2MD.Molecule.molecule import Molecule

class Trajectory(Molecule):
    """ Trajectory property class

        Parameters:          Type:
            keywords         dict        trajectory keyword list

        Attribute:           Type:
            gl_seed          int         global seed for random number generation
            initcond         int         sample initial condition or not
            excess           float       excess kinetic energy in Hartree
            scale            float       scaling factor of kinetic energy after applying excess kinetic energy
            target           float       target kinetic energy to scale up to
            graddesc         int         steepest descent using zero velocity molecular dynamics
            reset            int         reset initial velocity to remove translation and rotational velocity
            resetstep        int         frequency of resetting initial velocity 
            ninitcond        int         number of ensembles in initial condition sampling 
            method           str         initial condition sampling method
            format           str         frequence data format
            temp             float       temperature in Kelvin
            step             int         molecular dynamics steps
            size             float       molecular dynamics steps size in atomic time unit
          * ci               int         ci dimension, a.k.a. total number of states per spin multiplicity
          * nstate           int         number of electronic states
          * nnac             int         number of non-adiabatic coupling
          * nsoc             int         number of spin-orbit coupling
            root             int         root number, a.k.a. initial state
          * spin             int         total spin angular momentum per spin multiplicity
          * mult             list        multiplicity per spin
          * statemult        list        multiplicity per state
          * coupling         list        list of interstate coupling pairs
          * nac_coupling     list        list of non-adiabatic coupling pairs
          * soc_coupling     list        list of spin-orbit coupling pairs
            activestate      int         compute gradient only for active (current) state
            sfhp             str         surface hopping method
            nactype          str         nonadiabatic coupling approximation type
            phasecheck       int         nonadiabatic coupling phase correction based on the time overlap
            gap              float       energy gap threshold for Zhu-Nakamura surface hopping
            gapsoc           float       energy gap threshold for Zhu-Nakamura intersystem crossing
            delt             float       step size for fewest switches surface hopping 
            substep          int         number of substeps for fewest switches surface hopping
            integrate        int         integrate surface hopping probabliy in accumulation scheme (Not recommendated)
            deco             float       decoherence correction in Hartree
            adjust           int         adjust velocity at surface hopping
            reflect          int         reflect velocity at frastrated surface hopping
            maxh             int         maximum number of states that can hopping to
            dosoc            int         compute intersystem crossing
            thermo           str         choose an ensemble to apply thermostat or not
            thermodelay      int         delay time step for applying thermostat
            A                ndarray     the present state denesity matrix
            H                ndarray     the present energy matrix
            D                ndarray     the present non-adiabatic matrix
            last_A           ndarray     the previous state denesity matrix
            last_H           ndarray     the previous energy matrix
            last_D           ndarray     the previous non-adiabatic matrix
            last_nac         ndarray     the previous nonadiabatic coupling
            last_soc         ndarray     the previous spin-orbit coupling
            last_state       int         the previous state
            state            int         the present state
          * velo             ndarray     velocity in Bohr/atomic time unit
          * coord            ndarray     nonadiabatic nuclear coordinates in the present step  
            coord1           ndarray     nonadiabatic nuclear coordinates in one step before
       	    coord2           ndarray     nonadiabatic nuclear coordinates in two step before
          * energy           ndarray     potential energy in the present step
            energy1          ndarray     potential energy in one step before
            energy2          ndarray     potential energy in two step before
          * grad             ndarray     gradient in the present step
       	    grad1            ndarray     gradient in one step before
       	    grad2            ndarray     gradient in two step before
          * kinetic          float       kinetic energy in the present step
       	    kinetic1         float       kinetic energy in one step before
       	    kinetic2         float       kinetic energy in two step before
          * nac              ndarray     non-adibatic coupling vectors in Hartree/Bohr (numerator)
          * soc              ndarray     spin-orbit coupling in cm-1
            Vs               list        additional velocity information for thermostat array
            iter             int         current iteration
            iter_x           int         the last iteration in the excited state
            hoped            int         surface hopping type
            history          list        md history
            length           int         length of md history
          * status           int         molecular property calculation status
            verbose          int         verbose level of output information
            shinfo           str         surface hopping information

        Functions:           Returns:
            record           self        record the latest trajectory snapshot
            update_nu        self        update previous-previous and previous nuclear properties
            update_el        self        update previous-previous and previous electronic properties

    """

    __slots__ = ['gl_seed', 'initcond', 'excess', 'scale', 'target', 'graddesc', 'reset', 'resetstep',
                 'ninitcond', 'method', 'format', 'temp', 'step', 'size', 'root', 'attr', 'verbose', 'phasecheck',
                 'sfhp', 'gap', 'gapsoc', 'substep', 'integrate', 'deco', 'adjust', 'reflect', 'maxh', 'delt',
                 'last_state', 'state', 'last_A', 'last_H', 'last_D', 'A', 'H', 'D', 'dosoc', 'last_nac','last_soc',
                 'coord1', 'coord2', 'kinetic1', 'kinetic2', 'energy1', 'energy2', 'grad1', 'grad2', 'activestate',
                 'thermo', 'thermodelay', 'Vs', 'iter', 'iter_x', 'hoped', 'history', 'length', 'shinfo', 'nactype']

    def	__init__(self, mol, keywords = None):
        super().__init__(mol, keywords = keywords)
        self.attr = super().__slots__ + self.__slots__
        key_dict = keywords['md'].copy()

        ## load variables from key_dict
        self.gl_seed     = key_dict['gl_seed']
        self.initcond    = key_dict['initcond']
        self.excess      = key_dict['excess']
        self.scale       = key_dict['scale']
        self.target      = key_dict['target']
        self.graddesc    = key_dict['graddesc']
        self.reset       = key_dict['reset']
        self.resetstep   = key_dict['resetstep']
        self.ninitcond   = key_dict['ninitcond']
        self.method      = key_dict['method']
        self.format      = key_dict['format']
        self.temp        = key_dict['temp']
        self.step        = key_dict['step']
        self.size        = key_dict['size']
        self.root        = key_dict['root']
        self.activestate = key_dict['activestate']
        self.sfhp        = key_dict['sfhp']
        self.nactype     = key_dict['nactype']
        self.phasecheck  = key_dict['phasecheck']
        self.gap         = key_dict['gap']
        self.gapsoc      = key_dict['gapsoc']
        self.substep     = key_dict['substep']
        self.integrate   = key_dict['integrate']
        self.deco        = key_dict['deco']
        self.adjust      = key_dict['adjust']
        self.reflect     = key_dict['reflect']
        self.maxh        = key_dict['maxh']
        self.dosoc       = key_dict['dosoc']
        self.thermo      = key_dict['thermo']
        self.thermodelay = key_dict['thermodelay']
        self.length      = key_dict['record']
        self.last_state  = key_dict['root']
        self.state	 = key_dict['root']
        self.verbose     = key_dict['verbose']
        ## initialize variable for trajectory
        self.last_A      = np.zeros(0)
        self.last_H      = np.zeros(0)
        self.last_D      = np.zeros(0)
        self.last_nac    = np.zeros(0)
        self.last_soc    = np.zeros(0)
        self.A           = np.zeros(0)
        self.H           = np.zeros(0)
        self.D           = np.zeros(0)
        self.coord1      = np.zeros(0)
        self.coord2      = np.zeros(0)
        self.kinetic1    = np.zeros(0)
        self.kinetic2    = np.zeros(0)
        self.energy1     = np.zeros(0)
        self.energy2     = np.zeros(0)
        self.grad1       = np.zeros(0)
        self.grad2       = np.zeros(0)
        self.Vs          = []
        self.iter        = 0
        self.iter_x      = 0
        self.hoped       = 0
        self.history     = []
        self.delt        = 0.4134
        self.shinfo      = ''

        ## adjust step size for FSSH
        if self.substep == 0:
            self.substep = int(self.size / self.delt)
        else:
            self.delt = self.size / self.substep

        ## adjust nnac to skip reading nac results
        if self.nactype == 'ktdc':
            self.nnac = 0

    def record(self):
        ## do not record trajectory history
        if self.length == 0:
            return self

        ## record trajectory history
        self.history.append([self.iter,
                             self.state,
                             np.copy(self.atoms),
                             np.copy(self.coord),
                             np.copy(self.energy),
                             np.copy(self.grad),
                             np.copy(self.nac),
                             np.copy(self.soc),
                             np.copy(self.err_energy),
                             np.copy(self.err_grad),
                             np.copy(self.err_nac),
                             np.copy(self.err_soc),
                             np.copy(np.diag(np.real(self.A))),
                             ])

        ## keep the lastest steps of trajectories to save memory if the length is longer than requested
        if len(self.history) > self.length:
            self.history = self.history[-self.length:]

        return self

    def update_nu(self):
        # update previous-previous and previous nuclear properties
        self.coord2   = np.copy(self.coord1)
        self.coord1   = np.copy(self.coord)
        self.kinetic2 = self.kinetic1
        self.kinetic1 = self.kinetic
        self.energy2  = np.copy(self.energy1)
        self.energy1  = np.copy(self.energy)
        self.grad2    = np.copy(self.grad1)
        self.grad1    = np.copy(self.grad)
        self.last_soc = np.copy(self.soc)
        self.last_nac = self._phase_correction(self.last_nac, self.nac)

        return self

    def update_el(self):
        # update previous-previous and previous electronic properties
        self.last_A = np.copy(self.A)
        self.last_H = np.copy(self.H)
        self.last_D = np.copy(self.D)
        self.last_state = self.state

        return self

    def _phase_correction(self, ref, nac):
        ## nac phase correction based on time overlap
        if self.phasecheck == 0 or len(ref) == 0:
            cnac = np.copy(nac)

        else:
            cnac = []
            for n, d in enumerate(ref):
                f = np.sign(np.sum(ref * nac[n]))
                cnac.append(f * nac[n])
            cnac = np.array(cnac)

        return cnac
