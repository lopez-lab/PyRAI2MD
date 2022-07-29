######################################################
#
# PyRAI2MD 2 module for adaptive sampling
#
# Author Jingbai Li
# Sep 25 2021
#
######################################################

import multiprocessing, os, json, shutil, time
from multiprocessing import Pool
import numpy as np
import copy

from PyRAI2MD.methods import QM
from PyRAI2MD.Molecule.trajectory import Trajectory
from PyRAI2MD.Dynamics.aimd import AIMD
from PyRAI2MD.Machine_Learning.training_data import Data
from PyRAI2MD.Utils.coordinates import PrintCoord
from PyRAI2MD.Utils.aligngeom import AlignGeom
from PyRAI2MD.Utils.sampling import Sampling
from PyRAI2MD.Utils.bonds import BondLib
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong

class AdaptiveSampling:
    """ Adaptive sampling class

        Parameters:          Type:
            keywords         dict        keyword list

        Attribute:           Type:
            keywords         dict        keyword list
            title            str         calculation title
            data             class       training data class
            qm               str         quantum chemical method
            abinit           str         ab initio calculation method
            ml_ncpu          int         number of CPU for machine learning training
            qc_ncpu          int         number of CPU for quantum chemical calculation
            maxiter          int         maximum number of adaptive sampling iteration
            refine           int         refine the sampling at crossing region
            refine_num       int         number of refinement geometries
            refine_start     int         the starting MD step of refinement
            refine_end       int         the end MD step of refinement
            load             int         load a pre-trained model or train a model first
            transfer         int         transfer learning instead of fresh training
            pop_step         int         MD step cutoff for averaging state population
            maxsample        int         number of sampled geometries per trajectory
            dynsample        int         sample geometries using dyanmical error
            maxdiscard       int         maximum number of discarded snapshot for finding trouble geometries
            maxenergy        float       energy threshold to stop a trajectory
            minenergy        float       energy threshold to record a uncertain geometry
            dynenergy        float       factor to compute dynamical energy threshold between min and max
            inienergy        float       initial energy threshold
            fwdenergy        int         number of iteraction to delay raising dynamical energy error
            bckenergy        int         number of iteraction to delay reducing dynamical energy error
            maxgrad          float       gradient threshold to stop a trajectory
            mingrad          float       gradient threshold to record a uncertain geometry
            dyngrad          float       factor to compute dynamical gradient threshold between min and max
            inigrad          float       initial gradient threshold
            fwdgrad          int         number of iteraction to delay raising dynamical gradient error
            bckgrad          int         number of iteraction to delay reducing dynamical gradient error
            maxnac           float       nac threshold to stop a trajectory
            minnac           float       nac threshold to record a uncertain geometry
            dynnac           float       factor to compute dynamical nac threshold between min and max
            ininac           float       initial nac threshold
            fwdnac           int         number of iteraction to delay raising dynamical nac error
            bcknac           int         number of iteraction to delay reducing dynamical nac error
            maxsoc           float       soc threshold to stop a trajectory
            minsoc           float       soc threshold to record a uncertain geometry
            dynsoc           float       factor to compute dynamical soc threshold between min and max
            inisoc           float       initial soc threshold
            fwdsoc           int         number of iteraction to delay raising dynamical soc error
            bcksoc           int         number of iteraction to delay reducing dynamical soc error
            iter             int         number of search iteration
            ntraj            int         number of trajectories
            last             list        length per trajectory
            final            list        final state per trajectory
            geom             list        recorded geometries per trajectory
            energy           list        recorded energies per trajectory
            grad             list        recorded gradients per trajectory
            nac              list        recorded non-adibatic couplings per trajectory
            soc              list        recorded spin-orbit couplings per trajectory
            err_e            list        recorded energy errors per trajectory
            err_g            list        recorded gradient errors per trajectory
            err_n            list        recorded non-adibatic coupling errors per trajectory
            err_s            list        recorded spin-orbit coupling errors per trajectory
            pop              list        recorded state populations per trajectory
            max_e            float       maximum energy errors among all trajectories
            max_g            float       maximum gradient errors among all trajectories
            max_n            float       maximum non-adiabatic coupling errors among all trajectories
            max_s            float       maximum spin-orbit coupling errors among all trajectories
            dyn_e            list        list of dynamical energy error
            dyn_g            list        list of dynamical gradient error
            dyn_n            list        list of dynamical nac error
            dyn_s            list        list of dynamical soc error
            dyn_e_new        list        new list of dynamical energy error
            dyn_g_new        list        new list of dynamical gradient error
            dyn_n_new        list        new list of dynamical nac error
            dyn_s_new        list        new list of dynamical soc error
            itr_e            list        list of iteration delayed before adjusting dynamical energy error
            itr_g            list        list of iteration delayed before adjusting dynamical gradient error
            itr_n            list        list of iteration delayed before adjusting dynamical nac error
            itr_s            list        list of iteration delayed before adjusting dynamical soc error
            initcond         list        list of trajectory class for initial condition
            select_cond      list        list of trajectory class for selected geometries for QM calculation
            select_geom      list        list of coordinates for selected geometries
            nsampled         list        number of sampled geometries per trajectory
            nuncertain       list        number of uncertain geometries per trajectory
            nselect          list        number of selected geometries per trajectory
            ndiscard         list        number of discarded geometries per trajectory
            nrefine          list        number of refined geometries per trajectory

        Functions:           Returns:
            search           None        run adaptive sampling

    """

    def __init__(self, keywords = None):

        ## copy reusable information
        self.version      = keywords['version']
        self.keywords     = keywords.copy()

        if self.keywords['md']['record'] == 0:
            self.keywords['md']['record'] = 10

        ## load variables for generating initcond
        ninitcond         = keywords['md']['ninitcond']
        method            = keywords['md']['method']
        format            = keywords['md']['format']
        gl_seed           = keywords['md']['gl_seed']
        temp              = keywords['md']['temp']

        ## load variables for adaptive sampling
        self.title        = keywords['control']['title']
        self.qm           = keywords['control']['qm']
        self.abinit       = keywords['control']['abinit']
        self.ml_ncpu      = keywords['control']['ml_ncpu']
        self.qc_ncpu      = keywords['control']['qc_ncpu']
        self.maxiter      = keywords['control']['maxiter']
        self.refine       = keywords['control']['refine']
       	self.refine_num   = keywords['control']['refine_num']
       	self.refine_start = keywords['control']['refine_start']
       	self.refine_end   = keywords['control']['refine_end']
       	self.load         = keywords['control']['load']
        self.transfer     = keywords['control']['transfer']
        self.pop_step     = keywords['control']['pop_step']
        self.maxsample    = keywords['control']['maxsample']
        self.dynsample    = keywords['control']['dynsample']
        self.maxdiscard   = keywords['control']['maxdiscard']
        self.maxenergy    = keywords['control']['maxenergy']
        self.minenergy    = keywords['control']['minenergy']
        self.dynenergy    = keywords['control']['dynenergy']
        self.inienergy    = keywords['control']['inienergy']
        self.fwdenergy    = keywords['control']['fwdenergy']
        self.bckenergy    = keywords['control']['bckenergy']
        self.maxgrad      = keywords['control']['maxgrad']
        self.mingrad      = keywords['control']['mingrad']
        self.dyngrad      = keywords['control']['dyngrad']
        self.inigrad      = keywords['control']['inigrad']
        self.fwdgrad      = keywords['control']['fwdgrad']
        self.bckgrad      = keywords['control']['bckgrad']
        self.maxnac       = keywords['control']['maxnac']
        self.minnac       = keywords['control']['minnac']
        self.dynnac       = keywords['control']['dynnac']
        self.ininac       = keywords['control']['ininac']
        self.fwdnac       = keywords['control']['fwdnac']
        self.bcknac       = keywords['control']['bcknac']
        self.maxsoc       = keywords['control']['maxsoc']
        self.minsoc       = keywords['control']['minsoc']
        self.dynsoc       = keywords['control']['dynsoc']
        self.inisoc       = keywords['control']['inisoc']
        self.fwdsoc       = keywords['control']['fwdsoc']
        self.bcksoc       = keywords['control']['bcksoc']

        ## initialize trajectories stat
        self.iter         = 0
        self.ntraj        = ninitcond
        self.last         = []
        self.final        = []
        self.atoms        = []
        self.geom         = []
        self.energy       = []
        self.grad         = []
        self.nac          = []
        self.soc          = []
        self.err_e        = []
        self.err_g        = []
        self.err_n        = []
        self.err_s        = []
        self.pop          = []
        self.max_e        = 0
        self.max_g        = 0
        self.max_n        = 0
        self.max_s        = 0
        self.select_geom  = []
        self.nsampled     = []
        self.nuncertain   = []
        self.nselect      = []
        self.ndiscard     = []
        self.nrefine      = []
        self.select_cond  = []

       	## initialize dynamical	errors and delay steps
        if self.dynsample == 0:
            self.dyn_e = [self.maxenergy for x in range(ninitcond)]
       	    self.dyn_g = [self.maxgrad for x in range(ninitcond)]
       	    self.dyn_n = [self.maxnac for x in range(ninitcond)]
       	    self.dyn_s = [self.maxsoc for x in range(ninitcond)]
        else:        
            self.dyn_e = [self.minenergy + self.inienergy * (self.maxenergy - self.minenergy) for x in range(ninitcond)]
       	    self.dyn_g = [self.mingrad + self.inigrad * (self.maxgrad - self.mingrad) for x in range(ninitcond)]
       	    self.dyn_n = [self.minnac + self.ininac * (self.maxnac - self.minnac) for x in range(ninitcond)]
       	    self.dyn_s = [self.minsoc + self.inisoc * (self.maxsoc - self.minsoc) for x in range(ninitcond)]

        self.dyn_e_new = []
        self.dyn_g_new = []
        self.dyn_n_new = []
        self.dyn_s_new = []

       	self.itr_e = [0 for x in range(ninitcond)]
        self.itr_g = [0 for x in range(ninitcond)]
        self.itr_n = [0 for x in range(ninitcond)]
        self.itr_s = [0 for x in range(ninitcond)]

        self.itr_e_new = []
       	self.itr_g_new = []
       	self.itr_n_new = []
       	self.itr_s_new = []

        ## load training data
        train_data = keywords[self.qm]['train_data']
        self.data = Data()
        self.data.load(train_data)
        self.data.stat()

        ## generate initial conditions and trajectories
        np.random.seed(gl_seed)
        initcond = Sampling(self.title, ninitcond, gl_seed, temp, method, format)
        self.initcond = [Trajectory(x, keywords = self.keywords) for x in initcond]

        ## set multiprocessing
        multiprocessing.set_start_method('spawn')

    def _run_aimd(self):
        ## wrap variables for multiprocessing
        variables_wrapper = [[n, x] for n, x in enumerate(self.initcond)]
        ntraj = len(variables_wrapper)

        ## adjust multiprocessing if necessary
        ncpu = np.amin([ntraj, self.ml_ncpu])

        ## start multiprocessing
        md_traj = [[] for x in range(ntraj)]
        pool = multiprocessing.Pool(processes = ncpu)
        for val in pool.imap_unordered(self._aimd_wrapper, variables_wrapper):
            traj_id, md_hist = val
            md_traj[traj_id] = md_hist
        pool.close()

        return md_traj

    def _aimd_wrapper(self, variables):
        traj_id, traj = variables

        ## multiprocessing doesn't support shared-memory
        ## load mode in each worker process here :(
        qm = QM(self.qm, keywords = self.keywords, id = self.iter)
        qm.load()

        ## prepare AIMD
        aimd = AIMD(trajectory = traj,
                    keywords = self.keywords, 
                    qm = qm,
                    id = traj_id + 1,
                    dir = True)

        ## add dynamical errors
        aimd.maxerr_energy = self.dyn_e[traj_id]
        aimd.maxerr_grad   = self.dyn_g[traj_id]
        aimd.maxerr_nac    = self.dyn_n[traj_id]
        aimd.maxerr_soc    = self.dyn_s[traj_id]

        ## run AIMD
        md_traj = aimd.run()
        md_hist = md_traj.history
        return traj_id, md_hist

    def _run_abinit(self):
        ## wrap variables for multiprocessing
        variables_wrapper = [[n, x] for n, x in enumerate(self.select_cond)]
        ngeom = len(variables_wrapper)

        ## adjust multiprocessing if necessary
        ncpu = np.amin([ngeom, self.qc_ncpu])

        ## start multiprocessing
        qc_data = [[] for x in range(ngeom)]
        pool = multiprocessing.Pool(processes = ncpu)
        for val in pool.imap_unordered(self._abinit_wrapper, variables_wrapper):
            geom_id, xyz, energy, grad, nac, soc, completion = val
            qc_data[geom_id] = [[xyz, energy, grad, nac, soc], completion]
        pool.close()

        ## check qc results and exclude non-converged ones
        newdata=[[] for x in range(5)]
        for data, completion in qc_data:
            if  completion == 1:
                newdata[0].append(data[0])
                newdata[1].append(data[1])
                newdata[2].append(data[2])
                newdata[3].append(data[3])
                newdata[4].append(data[4])

        return newdata

    def _abinit_wrapper(self, variables):
        geom_id, mol = variables
        xyz = np.concatenate((self.atoms, mol.coord), axis = 1) 

        ## aling geometry to pre-stored training data to correct NAC phase
        ## it is not necessary if NAC is not request. The code below is obselete
        ## geom_pool = 
        ## choose = np.random.choice(np.arange(len(geom_pool)),np.amin([50,len(geom_pool)]),replace=False)
        ## geom_pool = np.array(geom_pool)[choose]
        ## similar, rmsd_min = AlignGeom(xyz, geom_pool)

        ## run QC calculation
        qc = QM(self.abinit, keywords = self.keywords, id = geom_id + 1)
        mol  = qc.evaluate(mol)

        ## prepare qc results
        energy     = mol.energy.tolist()
        grad       = mol.grad.tolist()
        nac        = mol.nac
        soc        = mol.soc
        completion = mol.status

        return geom_id, xyz, energy, grad, nac, soc, completion

    def _screen_error(self, md_traj):
        ## initialize data list

        ntraj          = len(md_traj)
        md_last        = [[] for x in range(ntraj)]
        md_final       = [[] for x in range(ntraj)]
        md_atoms       = [[] for x in range(ntraj)]
        md_geom        = [[] for x in range(ntraj)]
        md_energy      = [[] for x in range(ntraj)]
        md_grad        = [[] for x in range(ntraj)]
        md_nac         = [[] for x in range(ntraj)]
        md_soc         = [[] for x in range(ntraj)]
        md_err_e       = [[] for x in range(ntraj)]
        md_err_g       = [[] for x in range(ntraj)]
        md_err_n       = [[] for x in range(ntraj)]
        md_err_s       = [[] for x in range(ntraj)]
        md_dyn_e       = [[] for x in range(ntraj)]
        md_dyn_g       = [[] for x in range(ntraj)]
        md_dyn_n       = [[] for x in range(ntraj)]
        md_dyn_s       = [[] for x in range(ntraj)]
        md_itr_e       = [[] for x in range(ntraj)]
        md_itr_g       = [[] for x in range(ntraj)]
        md_itr_n       = [[] for x in range(ntraj)]
        md_itr_s       = [[] for x in range(ntraj)]
        md_pop         = [[] for x in range(ntraj)]
        md_refine_geom = [[] for x in range(ntraj)]
        md_select_geom = [[] for x in range(ntraj)]
        md_nsampled    = [[] for x in range(ntraj)]
        md_nuncertain  = [[] for x in range(ntraj)]
        md_nselect     = [[] for x in range(ntraj)]
        md_ndiscard    = [[] for x in range(ntraj)]
        md_nrefine     = [[] for x in range(ntraj)]
        md_select_cond = [[] for x in range(ntraj)]
        md_selec_e     = [[] for x in range(ntraj)]
       	md_selec_g     = [[] for x in range(ntraj)]
       	md_selec_n     = [[] for x in range(ntraj)]
       	md_selec_s     = [[] for x in range(ntraj)]

        ## screen trajectories with multiprocessing
        ncpu = np.amin([ntraj, self.ml_ncpu, 5])
        print('\nScreen geometries with %s CPUs' % (ncpu))
        t_s = time.time()

        n = 0
        variables_wrapper = [[n, x] for n, x in enumerate(md_traj)]
        pool = multiprocessing.Pool(processes = ncpu)
        for val in pool.imap_unordered(self._screen_error_wrapper, variables_wrapper):
        #for var in variables_wrapper:
            #val = self._screen_error_wrapper(var)
            traj_id, traj_data, sampling_data  = val
            md_last[traj_id]        = traj_data[0][-1]          # the trajectory length
            md_final[traj_id]       = traj_data[1][-1]          # the final state
            md_atoms[traj_id]       = traj_data[2][-1].tolist() # the atom list
            md_geom[traj_id]        = traj_data[3].tolist()     # all recorded geometries
            md_energy[traj_id]      = traj_data[4].tolist()     # all energies
            md_grad[traj_id]        = traj_data[5].tolist()     # all forces
            md_nac[traj_id]         = traj_data[6].tolist()     # all NACs
            md_soc[traj_id]         = traj_data[7].tolist()     # all SOCs
            md_err_e[traj_id]       = traj_data[8].tolist()     # all prediction error in energies
            md_err_g[traj_id]       = traj_data[9].tolist()     # all prediction error in forces
            md_err_n[traj_id]       = traj_data[10].tolist()    # all prediction error in NACs
            md_err_s[traj_id]       = traj_data[11].tolist()    # all prediction error in SOCs
            md_pop[traj_id]         = traj_data[12].tolist()    # all populations
            md_select_geom[traj_id] = sampling_data[0]          # selected geometries in each traj for qm calculations
            md_nsampled[traj_id]    = sampling_data[1]          # number of sampled geometries in each traj
            md_nuncertain[traj_id]  = sampling_data[2]          # number of uncertain traj
            md_nselect[traj_id]	    = sampling_data[3]          # number of selected geometries in each traj
            md_ndiscard[traj_id]    = sampling_data[4]          # number of discarded geometries in each traj
            md_nrefine[traj_id]     = sampling_data[5]          # number of refine geometries in each traj
            md_dyn_e[traj_id]       = sampling_data[6]          # all dynamical energy error
       	    md_dyn_g[traj_id]  	    = sampling_data[7] 	       	# all dynamical	grad error
       	    md_dyn_n[traj_id]  	    = sampling_data[8] 	       	# all dynamical	nac error
       	    md_dyn_s[traj_id]  	    = sampling_data[9]          # all dynamical	soc error
            md_itr_e[traj_id]       = sampling_data[10]         # all dynamical energy error delay
            md_itr_g[traj_id]       = sampling_data[11]         # all dynamical grad error delay
            md_itr_n[traj_id]       = sampling_data[12]         # all dynamical nac error delay
            md_itr_s[traj_id]       = sampling_data[13]         # all dynamical soc error delay
            md_selec_e[traj_id]     = sampling_data[14]         # all select energy error
       	    md_selec_g[traj_id]	    = sampling_data[15]	       	# all select gradient error
       	    md_selec_n[traj_id]	    = sampling_data[16]	       	# all select nac error
       	    md_selec_s[traj_id]	    = sampling_data[17]	       	# all select soc error
            n += 1
            print('Done %s of %s' % (n, ntraj))

        t_m = time.time()
        print('Select geometries spent:   ', HowLong(t_s, t_m))
        pool.close()

        ## find the max of data in different length
        md_max_e = np.amax([np.amax(x) for x in md_err_e])
        md_max_g = np.amax([np.amax(x) for x in md_err_g])
        md_max_n = np.amax([np.amax(x) for x in md_err_n])
        md_max_s = np.amax([np.amax(x) for x in md_err_s])

        ## update trajectories stat
        self.last       = md_last
        self.final      = md_final
        self.atoms      = md_atoms[-1]
        self.geom       = md_geom
        self.energy     = md_energy
        self.grad       = md_grad
        self.nac        = md_nac
        self.soc        = md_soc
        self.err_e      = md_err_e
        self.err_g      = md_err_g
        self.err_n      = md_err_n
        self.err_s      = md_err_s
        self.pop        = md_pop
        self.max_e      = md_max_e
        self.max_g      = md_max_g
        self.max_n      = md_max_n
        self.max_s      = md_max_s
        self.dyn_e_new  = md_dyn_e
        self.dyn_g_new  = md_dyn_g
        self.dyn_n_new  = md_dyn_n
        self.dyn_s_new  = md_dyn_s
        self.itr_e_new  = md_itr_e
       	self.itr_g_new  = md_itr_g
       	self.itr_n_new  = md_itr_n
       	self.itr_s_new  = md_itr_s
        self.nsampled   = md_nsampled
        self.nuncertain = md_nuncertain
        self.nselect    = md_nselect
        self.ndiscard   = md_ndiscard
        self.nrefine    = md_nrefine

        ## append selected geom and conditions
        self.select_geom = []
        self.select_cond = []

        for n, geom in enumerate(md_select_geom):
            self.select_geom = self.select_geom + geom

            for geo in geom:
                cond = copy.deepcopy(self.initcond[n])
                cond.coord = np.array(geo)
                self.select_cond.append(cond)

        t_e = time.time()
        print('Prepare calculation spent: ', HowLong(t_m, t_e))

        return self

    def _screen_error_wrapper(self, variables):
        ## This function screens errors from trajectories
        traj_id, traj = variables
        traj_data = np.array(traj).T
        sampling_data = [[] for x in range(18)]

        ## upack traj
        iter, state, atoms, geom, energy, grad, nac, soc, err_e, err_g, err_n, err_s, pop = traj_data
        allatoms = atoms[-1].tolist()

        ## find current dynamical errors and delay steps
        dyn_e = self.dyn_e[traj_id]
       	dyn_g =	self.dyn_g[traj_id]
       	dyn_n =	self.dyn_n[traj_id]
       	dyn_s =	self.dyn_s[traj_id]
        itr_e = self.itr_e[traj_id]
       	itr_g =	self.itr_g[traj_id]
       	itr_n =	self.itr_n[traj_id]
       	itr_s =	self.itr_s[traj_id]

        ## find index of geometries exceeding the threshold of prediction error
        index_e = self._sort_errors(err_e, self.minenergy)
        index_g = self._sort_errors(err_g, self.mingrad)
        index_n = self._sort_errors(err_n, self.minnac)
        index_s = self._sort_errors(err_s, self.minsoc)

        ## merge index and remove duplicate in select_geom
        index_tot = np.concatenate((index_e, index_g)).astype(int)
        index_tot = np.concatenate((index_tot, index_n)).astype(int)
        index_tot = np.concatenate((index_tot, index_s)).astype(int)
        index_tot = np.unique(index_tot)[::-1] # reverse from large to small indices
        num_index_tot = len(index_tot)

        ## find index of geometries exceeding the dynmical threshold of prediction error
        uncer_e = np.where(err_e > dyn_e)[0]
        uncer_g = np.where(err_g > dyn_g)[0]
        uncer_n = np.where(err_n > dyn_n)[0]
        uncer_s = np.where(err_s > dyn_s)[0]

        ## merge index and remove duplicate in uncertain trajectories
        uncer_tot = np.concatenate((uncer_e, uncer_g)).astype(int)
        uncer_tot = np.concatenate((uncer_tot, uncer_n)).astype(int)
        uncer_tot = np.concatenate((uncer_tot, uncer_s)).astype(int)
        uncer_tot = np.unique(uncer_tot)
        num_uncer_tot = len(uncer_tot)

        ## filter out the unphyiscal geometries based on atom distances
        if num_uncer_tot == 0:
            select_geom      = []
            num_select_geom  = 0
            num_discard_geom = 0
            dyn_e_new        = dyn_e
       	    dyn_g_new        = dyn_g
       	    dyn_n_new        = dyn_n
       	    dyn_s_new        = dyn_s
            itr_e_new        = itr_e
       	    itr_g_new  	     = itr_g
       	    itr_n_new  	     = itr_n
       	    itr_s_new  	     = itr_s
            selec_e          = err_e[index_e]
            selec_g          = err_g[index_g]
            selec_n          = err_n[index_n]
            selec_s          = err_s[index_s]

        else:
            select_geom = np.array(geom)[index_tot]
            select_geom, discard_geom, select_indx, discard_indx = self._distance_filter(allatoms, select_geom)
            select_geom = select_geom[0: self.maxsample]
            num_select_geom = len(select_geom)
            num_discard_geom = len(discard_geom)
            ndiscard_e = 0
            ndiscard_g = 0
            ndiscard_n = 0
            ndiscard_s = 0

            ## find number of discarded snapshot
            for indx in discard_indx:
                snapshot_indx = index_tot[indx]

                if snapshot_indx in index_e:
                    ndiscard_e += 1

       	       	if snapshot_indx in index_g:
       	       	    ndiscard_g += 1

       	       	if snapshot_indx in index_n:
       	       	    ndiscard_n += 1

       	       	if snapshot_indx in index_s:
       	       	    ndiscard_s += 1

            ## find error of selected snapshot
            ## initialize with zero to avoid empty list
            selec_e = [0]
            selec_g = [0]
            selec_n = [0]
            selec_s = [0]

            for indx in select_indx:
                snapshot_indx = index_tot[indx]

                if snapshot_indx in index_e:
                    selec_e.append(err_e[snapshot_indx])

                if snapshot_indx in index_g:
                    selec_g.append(err_g[snapshot_indx])

                if snapshot_indx in index_n:
                    selec_n.append(err_n[snapshot_indx])

                if snapshot_indx in index_s:
                    selec_s.append(err_s[snapshot_indx])

            ## update dynamical error
            dyn_e_new, itr_e_new = self._dynamical_error(dyn_e, self.dynenergy, self.minenergy, self.maxenergy, itr_e, self.fwdenergy, self.bckenergy, ndiscard_e, selec_e)
            dyn_g_new, itr_g_new = self._dynamical_error(dyn_g, self.dyngrad, self.mingrad, self.maxgrad, itr_g, self.fwdgrad, self.bckgrad, ndiscard_g, selec_g)
            dyn_n_new, itr_n_new = self._dynamical_error(dyn_n, self.dynnac, self.minnac, self.maxnac, itr_n, self.fwdnac, self.bcknac, ndiscard_n, selec_n)
            dyn_s_new, itr_s_new = self._dynamical_error(dyn_s, self.dynsoc, self.minsoc, self.maxsoc, itr_s, self.fwdsoc, self.bcksoc, ndiscard_s, selec_s)

        ## refine crossing region, optionally
        if self.refine == 1:
            energy = np.array([np.array(x) for x in energy])
            state = len(energy[0])
            pair = int(state * (state - 1) / 2)
            gap_e = np.zeros([len(energy), pair])  # initialize gap matrix
            pos = -1
            for i in range(state):                 # compute gap per pair of states
                for j in range(i + 1, state):
                    pos += 1
                    gap_e[:, pos] = np.abs(energy[:, i] - energy[:, j])
            gap_e = np.amin(gap_e, axis = 1)       # pick the smallest gap per point
            index_r = np.argsort(gap_e[self.refine_start: self.refine_end])

            refine_geom = np.array(geom)[index_r]
            refine_geom, refine_discard, refine_indx, refine_discard_indx = self._distance_filter(allatoms, refine_geom)
            refine_geom = refine_geom[0: self.refine_num]
            num_refine_geom = len(refine_geom)
        else:
            index_r = []
            refine_geom = []
            num_refine_geom = 0

        ## combine select and refine geom
        select_geom       = select_geom + refine_geom
        sampling_data[0]  = [x.tolist() for x in select_geom]  # to md_select_geom
        sampling_data[1]  = num_index_tot                      # to md_nsampled
        sampling_data[2]  = num_uncer_tot                      # to md_nuncertain
        sampling_data[3]  = num_select_geom                    # to md_nselect
        sampling_data[4]  = num_discard_geom                   # to md_ndiscard
        sampling_data[5]  = num_refine_geom                    # to md_nrefine
        sampling_data[6]  = dyn_e_new                          # to md_dyn_e
       	sampling_data[7]  = dyn_g_new                          # to md_dyn_g
       	sampling_data[8]  = dyn_n_new                          # to md_dyn_n
       	sampling_data[9]  = dyn_s_new                          # to md_dyn_s
        sampling_data[10] = itr_e_new                          # to md_itr_e
       	sampling_data[11] = itr_g_new  	                       # to md_itr_g
       	sampling_data[12] = itr_n_new                          # to md_itr_n
       	sampling_data[13] = itr_s_new                          # to md_itr_s
        sampling_data[14] = selec_e                            # to md_selec_e
        sampling_data[15] = selec_g   	       	       	       # to md_selec_g
        sampling_data[16] = selec_n   	       	       	       # to md_selec_n
        sampling_data[17] = selec_s   	       	       	       # to md_selec_s

        return traj_id, traj_data, sampling_data

    def _distance_filter(self, atom, geom):
        ## This function filter out unphysical geometries based on atom distances
        keep = []
        discard = []
        keep_indx = []
        discard_indx = []
        if len(geom) > 0:
            natom = len(geom[0])
            for n, geo in enumerate(geom):
                unphysical = 0
                for i in range(natom):
                    for j in range(i + 1, natom): 
                        atom1 = atom[i][0]
                        coord1 = np.array(geo[i][0: 3])
       	       	       	atom2 = atom[j][0]
                        coord2 = np.array(geo[j][0: 3])
                        distance = np.sum((coord1 - coord2)**2)**0.5
                        threshld = BondLib(atom1, atom2)
                        if distance < threshld * 0.7:
                            #print(i+1, atom1, coord1, j+1, atom2, coord2, distance, threshld, threshld * 0.7)
                            unphysical=1
                            break 
                if unphysical == 1:
                    discard.append(geo) 
                    discard_indx.append(n)
                else:
                    keep.append(geo)
                    keep_indx.append(n)

        return keep, discard, keep_indx, discard_indx

    def _sort_errors(self, err, threshold):
        ## This function sort the errors from the largest to the smallest
        ## here the max error is the dynamical error

        sort_i_err = np.argsort(-err)
        sort_err  = err[sort_i_err]
        find_err  = np.argwhere(sort_err > threshold)
        index_err = sort_i_err[find_err]

        return index_err

    def _dynamical_error(self, dyn_err, delt_err, min_err, max_err, itr_err, fwd_delay, bck_delay, ndiscard, err):
        ## This function adjust the dynamical error
        if   ndiscard > self.maxdiscard:
            itr = itr_err - 1

        elif ndiscard <= self.maxdiscard and np.amax(err) > dyn_err:
            itr = itr_err + 1

        else:
            itr = itr_err

        if   itr == fwd_delay:
            dyn_err_new = np.amin([dyn_err + delt_err * (max_err - min_err), max_err])
            itr_err_new = 0

        elif itr == -bck_delay:
            dyn_err_new = np.amax([dyn_err - delt_err * (max_err - min_err), min_err])
            itr_err_new = 0

        else:
            dyn_err_new = dyn_err
            itr_err_new = itr

        return dyn_err_new, itr_err_new

    def _localmax(self, error, maxsample, neighbor):
        ## This function find local maximum of error as function of simulation step
        ## find all local maximum: g_1 gradient from n-1 to n; g_2 gradient from n to n+1
        index_lcmax = []   # index for local maximum
        error_lcmax = []   # local maximum errors
        for n, i in enumerate(error):
            if i == error[0]:
                g_1 = 1
            else:
                g_1 = i-error[n - 1]

            if i == error[-1]:
                g_2 = 0
            else:
                g_2 = error[n + 1]-i

            if g_1 >0 and g_2 <= 0:
                index_lcmax.append(n)
                error_lcmax.append(i)

        index_lcmax = np.array(index_lcmax)
        error_lcmax = np.array(error_lcmax)
        check_lcmax = np.ones(len(error_lcmax))

        ## only include non-neighbor
        index_error = []
        selec_error = []
        for n, i in enumerate(np.argsort(-error_lcmax)):
            if check_lcmax[i] == 1:
                index_error.append(index_lcmax[i])
                selec_error.append(error_lcmax[i])
                for j in np.argsort(-error_lcmax)[n + 1:]:
                    if np.abs(index_lcmax[i] - index_lcmax[j]) < neighbor:
                        check_lcmax[j] = 0

        index_error = np.array(index_error)
        selec_error = np.array(selec_error)
        ## adjust maxsample
        if len(selec_error) > maxsample:
            selec_error = selec_error[: maxsample]
            index_error = index_error[: maxsample]

        return selec_error, index_error

    def _update_dynamical_error(self):
        self.dyn_e = copy.deepcopy(self.dyn_e_new)
        self.dyn_g = copy.deepcopy(self.dyn_g_new)
        self.dyn_n = copy.deepcopy(self.dyn_n_new)
        self.dyn_s = copy.deepcopy(self.dyn_s_new)

        self.itr_e = copy.deepcopy(self.itr_e_new)
        self.itr_g = copy.deepcopy(self.itr_g_new)
        self.itr_n = copy.deepcopy(self.itr_n_new)
        self.itr_s = copy.deepcopy(self.itr_s_new)

        return self

    def _update_train_set(self, newdata):
        self.data.append(newdata)
        self.data.stat()
        self.data.save(self.iter + 1)

        return self

    def _train_model(self):
        ## add training data to keywords
        self.keywords[self.qm]['train_mode'] = 'training'
        self.keywords[self.qm]['data'] = self.data

        ## copy NN weights for transfer learning
        if self.iter == 2 and self.transfer == 1:
            self.keywords[self.qm]['train_mode'] = 'retraining'
            shutil.copytree('NN-%s' % (self.title), 'NN-%s-%s' % (self.title, self.iter))
 
        if self.iter > 2 and self.transfer == 1:
            self.keywords[self.qm]['train_mode'] = 'retraining'
            shutil.copytree('NN-%s-%s' % (self.title, self.iter-1), 'NN-%s-%s' % (self.title, self.iter))

        ## start NN training
        if self.iter > 1 or self.load == 0:
            pool = multiprocessing.Pool(processes=1)
            for val in pool.imap_unordered(self._train_wrapper, [None]):
                val = None
            pool.close()
 
        return self

    def _train_wrapper(self, fake):
        model = QM(self.qm, keywords = self.keywords, id = self.iter)
        model.train()

        return self

    def _checkpoint(self):
        logpath      = os.getcwd()
        completed    = 0
        ground       = 0
        traj_info = '  &adaptive sampling progress\n'
        for i in range(self.ntraj):
            marker = ''

            if self.nuncertain[i] == 0:
                completed += 1
                marker = '*'

            if self.final[i] == 1:
                ground += 1

            del_dyn_e = self.dyn_e_new[i] - self.dyn_e[i]
       	    del_dyn_g = self.dyn_g_new[i] - self.dyn_g[i]
       	    del_dyn_n = self.dyn_n_new[i] - self.dyn_n[i]
       	    del_dyn_s = self.dyn_s_new[i] - self.dyn_s[i]

            if   del_dyn_e > 0:
                marker_e = '>>>'

            elif del_dyn_e < 0:
       	       	marker_e = '<<<'

            else:
       	       	marker_e = '==='
            
            if   del_dyn_g > 0:
       	       	marker_g = '>>>'

       	    elif del_dyn_g < 0:
                marker_g = '<<<'

       	    else:
                marker_g = '==='

            if   del_dyn_n > 0:
       	       	marker_n = '>>>'

       	    elif del_dyn_n < 0:
                marker_n = '<<<'

       	    else:
                marker_n = '==='

            if   del_dyn_s > 0:
       	       	marker_s = '>>>'

       	    elif del_dyn_s < 0:
                marker_s = '<<<'

       	    else:
                marker_s = '==='

            traj_info += '  Traj(ID: %6s Step: %8s State: %2s) =>\
Geom(Sampled: %3s Collect: %3s Discard: %3s Refine: %s) =>\
MaxStd(Energy: %8.4f Gradient: %8.4f NAC: %8.4f SOC: %8.4f) =>\
Thrshd(E: %8.4f %s %8.4f G: %8.4f %s %8.4f N: %8.4f %s %8.4f S: %8.4f %s %8.4f) %s\n' % (
                i + 1,
                self.last[i], 
                self.final[i],
                self.nsampled[i],
                self.nselect[i],
                self.ndiscard[i],
                self.nrefine[i],
                np.amax(self.err_e[i]),
                np.amax(self.err_g[i]),
                np.amax(self.err_n[i]),
                np.amax(self.err_s[i]),
                self.dyn_e[i],
                marker_e,
                self.dyn_e_new[i],
                self.dyn_g[i],
       	       	marker_g,
       	       	self.dyn_g_new[i],
                self.dyn_n[i],
       	       	marker_n,
       	       	self.dyn_n_new[i],
                self.dyn_s[i],
       	       	marker_s,
       	       	self.dyn_s_new[i],
                marker)

        log_info="""
%s

  &adaptive sampling iter %5s
-------------------------------------------------------
  Number of trajectories:     %-10s
  Average length:             %-10s
  Ground state:               %-10s
  Completed:                  %-10s
  Sampled trajectories:       %-10s
  Sampled geometries          %-10s
  Collected:                  %-10s
  Discarded:                  %-10s
  Refinement:                 %-10s

  Metrics                     MaxStd  Threshold    Pass
  Energy:                   %8.4f   %8.4f  %6s
  Gradient:                 %8.4f   %8.4f  %6s
  Non-adiabatic coupling:   %8.4f   %8.4f  %6s
  Spin-orbit coupling:      %8.4f   %8.4f  %6s
-------------------------------------------------------
"""  % (traj_info, 
        self.iter,
        self.ntraj,
        np.mean(self.last),
        ground,
        completed,
        np.sum(self.nuncertain),
        np.sum(self.nsampled),
        np.sum(self.nselect),
        np.sum(self.ndiscard),
        np.sum(self.nrefine),
        self.max_e,
        np.mean(self.dyn_e),
        self.max_e <= np.mean(self.dyn_e),
        self.max_g,
        np.mean(self.dyn_g),
        self.max_g <= np.mean(self.dyn_g),
        self.max_n,
        np.mean(self.dyn_n), 
        self.max_n <= np.mean(self.dyn_n),
        self.max_s,
        np.mean(self.dyn_s),
        self.max_s <= np.mean(self.dyn_s))

        print(log_info)
        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(log_info)

        ## save state population info
        average_pop=[]
        for p in self.pop:
            if len(p) >= self.pop_step:
                average_pop.append(p[0: self.pop_step])
        if len(average_pop) > 0:
            average_pop = np.mean(average_pop, axis=0)
            pop_info=''
            for n, p in enumerate(average_pop):
                pop_info += '%-5s%s\n'% (n,' '.join(['%24.16f ' % (x) for x in p]))
            with open('%s/%s-%s.pop' % (logpath, self.title, self.iter), 'w') as pop:
                pop.write(pop_info)

        ## save the latest geometries:
        geom_coord = ''
        for i in range(self.ntraj):
            last_step = self.last[i]
            last_state = self.final[i] 
            last_geom = self.geom[i][-1]
            natom = len(self.atoms)
            cmmt = 'id %s coord %d state %d' % (i + 1, last_step, last_state)
            geom_coord += '%s\n%s\n%s\n' % (natom, cmmt, PrintCoord(np.concatenate((self.atoms, last_geom), axis = 1)))
        with open('%s/%s-ntraj-%s-iter-%s.xyz' % (logpath, self.title, self.ntraj, self.iter), 'w') as coord:
                coord.write(geom_coord)

        ## save adaptive sampling results
        #savethis = {self.iter:checkpoint_dict} ## This saves too much !!
        #print(self.select_geom)
        savethis = {
            self.iter: {
                'ntraj'     : self.ntraj,
                'length'    : float(np.mean(self.last)),
                'ground'    : ground,
                'completed' : completed,
                'uncertain' : int(np.sum(self.nuncertain)),
                'refine'    : int(np.sum(self.nrefine)),
                'select'    : int(np.sum(self.nselect)),
                'discard'   : int(np.sum(self.ndiscard)),
                'atoms'     : self.atoms,
                'new_geom'  : self.select_geom,
            }
        }
        if self.iter == 1:
            with open('%s/%s.adaptive.json' % (logpath,self.title), 'w') as outfile:
                json.dump(savethis, outfile)
        else:
            with open('%s/%s.adaptive.json' % (logpath,self.title), 'r') as infile:
                loadthis = json.load(infile)
            savethis.update(loadthis)
            with open('%s/%s.adaptive.json' % (logpath,self.title), 'w') as outfile:
                json.dump(savethis,outfile)

        self.completed = completed
        return completed

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |          Adaptive Sampling for ML-NAMD            |
 |                                                   |
 *---------------------------------------------------*


 Number of iterations:  %s

""" % ( self.version, 
        self.maxiter)

        return headline

    def search(self):
        logpath = os.getcwd()
        start = time.time()
        heading = 'Adaptive Sampling Start: %20s\n%s' % (WhatIsTime(), self._heading())
        print(heading)
        with open('%s/%s.log' % (logpath, self.title), 'w') as log:
            log.write(heading)

        for iter in range(self.maxiter):
            self.iter = iter + 1
            self._train_model()
            md_traj = self._run_aimd()
            self._screen_error(md_traj)
            completed = self._checkpoint()

            if self.iter > self.maxiter:
                break

            if completed == self.ntraj and np.sum(self.nrefine) == 0:
                break

            newdata = self._run_abinit()
            self._update_train_set(newdata)

            if self.dynsample != 0:
                self._update_dynamical_error()

        end = time.time()
        walltime = HowLong(start, end)
        tailing = 'Adaptive Sampling End: %20s Total: %20s\n' % (WhatIsTime(), walltime)
        print(tailing)
        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(tailing)
