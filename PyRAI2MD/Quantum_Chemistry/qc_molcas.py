######################################################
#
# PyRAI2MD 2 module for MOLCAS interface
#
# Author Jingbai Li
# Sep 20 2021
#
######################################################

import os, sys, subprocess, shutil
import numpy as np

from PyRAI2MD.Utils.coordinates import PrintCoord, S2F, Markatom, MolcasCoord

class MOLCAS:
    """ MOLCAS single point calculation interface

        Parameters:          Type:
            keywords         dict         keywords dict
            id               int          calculation index

        Attribute:           Type:
            natom            int         number of atoms.
            nstate           int         number of electronic states
            nnac             int         number of non-adiabatic couplings
            nsoc             int         number of spin-orbit couplings
            state            int         the current state
            activestate      int         compute gradient only for the current state
            ci               list        number of state per spin multiplicity
            mult             list        spin multiplicity
            nac_coupling     list        non-adiabatic coupling pairs
            soc_coupling     list        spin-orbit coupling pairs
            keep_tmp         int  	 keep the Molcas calculation folders (1) or not (0).
            verbose          int	 print level.
            track_phase      int	 track NAC phase using CAS(2,2). Currently not used
            basis            int	 use customized basis set (1) or not(2).
            project          str	 calculation name.
            workdir          str	 Molcas workdir, molcas use this to write tmp files.
            calcdir          str	 calculation folder.
            molcas           str	 Molcas enviroment variable, executable folder.
            molcas_nproc     int	 Molcas_nproc enviroment variable,number of threads for OMP parallelization.
            molcas_mem       int	 Molcas_mem enviroment variable, calculation memory in MB.
            molcas_print     int	 Molcas_print enviroment variable, print level.
            threads          int	 number of threads for OMP parallelization.
            use_hpc          int	 use HPC (1) for calculation or not(0), like SLURM.

        Functions:           Returns:
            train            self        fake function
            load             self        fake function
            appendix         self        fake function
            evaluate         self        run single point calculation

    """

    def __init__(self, keywords = None, id = None, runtype = 'qm'):

        self.runtype        = runtype
        self.ci             = []
        self.mult           = []
        self.nstate         = 0
        self.nnac           = 0
        self.nsoc           = 0
        self.soc_coupling   = []
        self.state          = 0
        self.activestate    = 0
        variables           = keywords['molcas']
        self.keep_tmp       = variables['keep_tmp']
        self.verbose        = variables['verbose']
        self.track_phase    = variables['track_phase']
        self.basis          = variables['basis']
        self.project        = variables['molcas_project']
        self.workdir        = variables['molcas_workdir']
        self.calcdir        = variables['molcas_calcdir']
        self.molcas         = variables['molcas']
        self.molcas_nproc   = variables['molcas_nproc']
        self.molcas_mem     = variables['molcas_mem']
        self.molcas_print   = variables['molcas_print']
        self.threads        = variables['omp_num_threads']
        self.use_hpc        = variables['use_hpc']

        ## check calculation folder
        ## add index when running in adaptive sampling

        if   id != None:
            self.calcdir    = '%s/tmp_MOLCAS-%s' % (self.calcdir, id)

        elif id == 'Read':
            self.calcdir    = self.calcdir

        else:
            self.calcdir    = '%s/tmp_MOLCAS' % (self.calcdir)

        ## check Molcas workdir
        ## use local scratch /srv/tmp or /tmp if set to 'AUTO'

        if   self.workdir == 'AUTO':
            if os.path.exists('/srv/tmp') == True:
                self.workdir = '/srv/tmp/%s/%s/%s' % (
                    os.environ['USER'],
                    self.calcdir.split('/')[-2],
                    self.calcdir.split('/')[-1])

            else:
                self.workdir = '/tmp/%s/%s/%s' % (
                    os.environ['USER'],
                    self.calcdir.split('/')[-2],
                    self.calcdir.split('/')[-1])

        elif self.workdir == None:
            self.workdir = self.calcdir

        ## initialize runscript
        self.runscript = """
export MOLCAS=%s
export CALCDIR=%s
export MOLCAS_PROJECT=%s
export MOLCAS_WORKDIR=%s
export MOLCAS_NPROCS=%s
export MOLCAS_MEM=%s
export MOLCAS_PRINT=%s
export OMP_NUM_THREADS=%s
export PATH=$MOLCAS/bin:$PATH

cd $CALCDIR
mkdir -p $MOLCAS_WORKDIR/$MOLCAS_PROJECT
$MOLCAS/bin/pymolcas -f $MOLCAS_PROJECT.inp -b 1
rm -r $MOLCAS_WORKDIR/$MOLCAS_PROJECT
""" % (     self.molcas,
            self.calcdir,
            self.project,
            self.workdir,
            self.molcas_nproc,
            self.molcas_mem,
            self.molcas_print,
            self.threads)

    def _setup_hpc(self):
        ## setup calculation using HPC
        ## read slurm template from .slurm files

        if os.path.exists('%s.slurm' % (self.project)) == True:
            with open('%s.slurm' % (self.project)) as template:
                submission=template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  Molcas: looking for submission file %s.slurm' % (self.project))

        submission += '\n%s' % (self.runscript)

        with open('%s/%s.sbatch' % (self.calcdir, self.project), 'w') as out:
            out.write(submission)

    def _setup_molcas(self, x):
        ## read input template from current directory
        ## make calculation folder and input file
        with open('%s.molcas' % (self.project), 'r') as template:
            input = template.read()

        if os.path.exists(self.calcdir) == False:
            os.makedirs(self.calcdir)

        #if os.path.exists(self.workdir) == False:
        #    os.makedirs(self.workdir)

        if self.activestate == 1:
            input = input.split('&')
            si_input = []
            grad_pos = 1
            grad_root = 1
            for n, substate in enumerate(self.ci):
                if self.state > np.sum(self.ci[:n]) and self.state <= np.sum(self.ci[:n + 1]):
                    grad_pos = n + 1
                    grad_root = self.state - np.sum(self.ci[:n])
            section = 0
            for n, line in enumerate(input):
                if 'ALASKA' in line.upper() and 'ROOT' in line.upper():
                    continue
                else:
                    si_input.append(line)

       	       	if 'RASSCF' in line.upper():
       	       	    section += 1
                    if grad_pos == section:
                        si_input.append('ALASKA\nROOT=%d\n' % (grad_root))

            si_input = '&'.join(si_input)

            with open('%s/%s.inp' % (self.calcdir, self.project), 'w') as newinput:
                newinput.write(si_input)
        else:
            shutil.copy2('%s.molcas' % (self.project), '%s/%s.inp' % (self.calcdir, self.project))

        ## prepare .xyz .StrOrb files
        self._write_coord(x)

        if  os.path.exists('%s.StrOrb' % (self.project)) == True and\
            os.path.exists('%s/%s.RasOrb' % (self.calcdir, self.project)) == False:
            shutil.copy2('%s.StrOrb' % (self.project), '%s/%s.StrOrb' % (self.calcdir, self.project))
        elif os.path.exists('%s.StrOrb' % (self.project)) == True and\
            os.path.exists('%s/%s.RasOrb' % (self.calcdir, self.project)) == True:
            shutil.copy2('%s/%s.RasOrb' % (self.calcdir, self.project), '%s/%s.StrOrb' % (self.calcdir, self.project))
        elif os.path.exists('%s.JobIph' % (self.project)) == True and\
            os.path.exists('%s/%s.JobIph.new' % (self.calcdir, self.project)) == False:
            shutil.copy2('%s.JobIph' % (self.project), '%s/%s.JobIph' % (self.calcdir, self.project)) 
        elif os.path.exists('%s.JobIph' % (self.project)) == True and\
            os.path.exists('%s/%s.JobIph' % (self.calcdir, self.project)) == True:
           shutil.copy2('%s/%s.JobIph.new' % (self.calcdir, self.project), '%s/%s.JobIph' % (self.calcdir, self.project))
        else:
            sys.exit('\n  FileNotFoundError\n  Molcas: looking for guess orbital %s.StrOrb or %s.JobIph ' % (self.project, self.project))

        ## write run script
        with open('%s/%s.sh' % (self.calcdir, self.project), 'w') as out:
            out.write(self.runscript)

        ## setup HPC settings
        if self.use_hpc == 1:
            self._setup_hpc()


    def _mark_atoms(self, x):
        ## prepare a list for marking atoms with different basis sets if necessary
        marks = []
        if os.path.exists('%s.basis' % (self.project)) == True:
            with open('%s.basis' % (self.project)) as atommarks:
                marks = atommarks.read().splitlines()
                natom = int(marks[0])
                marks = marks[2: 2 + natom]

        if self.basis == 1 and len(marks)>0:
            x = Markatom(x, marks, 'molcas')

        return x

    def _write_coord(self, x):
        ## write coordinate file 
        natom = len(x)
        xyz = '%s\n\n%s' % (natom, PrintCoord(x))

        ## save xyz and orbital files
        with open('%s/%s.xyz' % (self.calcdir, self.project), 'w') as out:
            out.write(xyz)

    def _run_molcas(self):
        ## run molcas calculation

        maindir = os.getcwd()
        os.chdir(self.calcdir)
        if self.use_hpc == 1:
            subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.calcdir, self.project)])
        else:
            subprocess.run(['bash', '%s/%s.sh' % (self.calcdir, self.project)])
        os.chdir(maindir)

    def _read_data(self, natom):
        ## read molcas logfile and pack data

        if os.path.exists('%s/%s.log' % (self.calcdir, self.project)) == False:
            return [], np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

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
                g = log[i + 8: i + 8 + natom]
                g = S2F(g)
                gradient.append(g)

            elif """CI derivative coupling""" in line:
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
                socme = float(soc_mtx[s1, s2 - self.ci[0] + sin_state]) ## assume low spin is in front of high spin (maybe generalize later)
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

    def _qmmm(self, traj):
        ## run Molcas for QMMM calculation

        ## create qmmm model
        traj = traj.applyqmmm()

        xyz = np.concatenate((traj.qm_atoms, traj.qm_coord), axis=1)
        nxyz = len(xyz)

        ## mark atom if requested
        xyz = self._mark_atoms(xyz)

        ## setup Molcas calculation
        self._setup_molcas(xyz)

        ## run Molcas calculation
        self._run_molcas()

        ## read Molcas output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        ## project force and coupling
        jacob = traj.Hcap_jacob
        gradient = np.array([np.dot(x, jacob) for x in gradient])
        nac = np.array([np.dot(x, jacob) for x in nac])

       	return energy, gradient, nac, soc


    def _qm(self, traj):
        ## run Molcas for QM calculation 

        xyz = np.concatenate((traj.atoms, traj.coord), axis=1)
        nxyz = len(xyz)

        ## mark atom if requested
        xyz = self._mark_atoms(xyz)

        ## setup Molcas calculation
        self._setup_molcas(xyz)

        ## run Molcas calculation
        self._run_molcas()

        ## read Molcas output files
        coord, energy, gradient, nac, soc = self._read_data(nxyz)

        return energy, gradient, nac, soc

    def appendix(self, addons):
        ## fake function

        return self

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

        if   self.runtype == 'qm':
            energy, gradient, nac, soc = self._qm(traj)
        elif self.runtype == 'qmmm':
       	    energy, gradient, nac, soc = self._qmmm(traj)

        ## phase correction
        #if self.track_phase == 1:
        #    nac = self._phase_correction(nac, nac1)
        # add this function in the future if really needed

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

    def train(self):
        ## fake function

        return self

    def load(self):
        ## fake function

        return self
