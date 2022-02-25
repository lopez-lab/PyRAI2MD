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
            maxenergy'       float       energy threshold to stop a trajectory
            minenergy'       float       energy threshold to record a uncertain geometry
            maxgrad'         float       gradient threshold to stop a trajectory
            mingrad'         float       gradient threshold to record a uncertain geometry
            maxnac'          float       nac threshold to stop a trajectory
            minnac'          float       nac threshold to record a uncertain geometry
            maxsoc'          float       soc threshold to stop a trajectory
            minsoc'          float       soc threshold to record a uncertain geometry
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
            initcond         list        list of trajectory class for initial condition
            select_cond      list        list of trajectory class for selected geometries for QM calculation
            select_geom      list        list of coordinates for selected geometries
            nselect          list        number of selected geometries per trajectory
            ndiscard         list        number of discarded geometries per trajectory
            nuncertain       list        number of uncertain geometries per trajectory
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
        self.maxenergy    = keywords['control']['maxenergy']
        self.minenergy    = keywords['control']['minenergy']
        self.maxgrad      = keywords['control']['maxgrad']
        self.mingrad      = keywords['control']['mingrad']
        self.maxnac       = keywords['control']['maxnac']
        self.minnac       = keywords['control']['minnac']
        self.maxsoc       = keywords['control']['maxsoc']
        self.minsoc       = keywords['control']['minsoc']

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
        self.nselect      = []
        self.ndiscard     = []
        self.nuncertain   = []
        self.nrefine      = []
        self.select_cond  = []

        ## load training data
        train_data = keywords[self.qm]['train_data']
        self.data = Data()
        self.data.load(train_data)
        self.data.stat()

        ## generate initial conditions and trajectories
        np.random.seed(gl_seed)
        initcond = Sampling(self.title, ninitcond, gl_seed, temp, method, format)
        self.initcond = [Trajectory(x, keywords = self.keywords) for x in initcond]

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

        ## run AIMD
        aimd = AIMD(trajectory = traj,
                    keywords = self.keywords, 
                    qm = qm,
                    id = traj_id + 1,
                    dir = True)
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
        ## initialize info list
        md_last = []
        md_final = []
        md_geom = []
        md_energy = []
        md_grad = []
        md_nac = []
        md_soc = []
        md_err_e = []
        md_err_g = []
        md_err_n = []
        md_err_s = []
        md_pop = []
        md_max_e = 0
       	md_max_g = 0
       	md_max_n = 0
        md_max_s = 0
        md_select_geom = []
        md_nselect = []
        md_ndiscard = []
        md_uncertain = []
        md_nrefine = []
        self.select_cond = []

        ## screen trajectories
        for traj_id, traj in enumerate(md_traj):
            iter, state, atoms, geom, energy, grad, nac, soc, err_e, err_g, err_n, err_s, pop = np.array(traj).T
            last = iter[-1]
            final = state[-1]
            allatoms = atoms[-1].tolist()
            ## pack data into checkpoing dict
            md_last.append(last)              # the trajectory length
            md_final.append(final)              # the final state
            md_geom.append(geom.tolist())     # all recorded geometries
            md_energy.append(energy.tolist()) # all energies
            md_grad.append(grad.tolist())     # all forces
            md_nac.append(nac.tolist())       # all NACs
            md_soc.append(soc.tolist())       # all SOCs
            md_err_e.append(err_e.tolist())   # all prediction error in energies
       	    md_err_g.append(err_g.tolist())   # all prediction error in forces
       	    md_err_n.append(err_n.tolist())   # all prediction error in NACs
            md_err_s.append(err_s.tolist())   # all prediction error in SOCs
            md_pop.append(pop.tolist())       # all populations

            ## update the maximum errors
            if np.amax(err_e) > md_max_e:
                md_max_e = np.amax(err_e)     # max prediction error in energies
            if np.amax(err_g) > md_max_g:
                md_max_g = np.amax(err_g)     # max prediction error in forces
            if np.amax(err_n) > md_max_n:
      	       	md_max_n = np.amax(err_n)     # max prediction error in NACs
            if np.amax(err_s) > md_max_s:
                md_max_s = np.amax(err_s)     # max prediction error in SOCs

            ## largest n std in e, g, and n
            #selec_e,index_e = self._localmax(err_e,maxsample,neighbor)
            #selec_g,index_g = self._localmax(err_g,maxsample,neighbor)
            #selec_n,index_n = self._localmax(err_n,maxsample,neighbor)

            ## find index of geometries exceeding the threshold of prediction error
            index_e = self._sort_errors(err_e, self.minenergy)
            selec_e = err_e[index_e]

            index_g = self._sort_errors(err_g, self.mingrad)
            selec_g = err_g[index_g]

            index_n = self._sort_errors(err_n, self.minnac)
            selec_n = err_n[index_n]

            index_s = self._sort_errors(err_s, self.minsoc)
            selec_s = err_s[index_s]

            ## find index of geometries exceeding the max threshold of prediction error
            uncer_e = self._sort_errors(err_e, self.maxenergy)
            uncer_g = self._sort_errors(err_g, self.maxgrad)
            uncer_n = self._sort_errors(err_n, self.maxnac)
            uncer_s = self._sort_errors(err_s, self.maxsoc)

            ##  merge index and remove duplicate in select_geom
            index_tot = np.concatenate((index_e, index_g)).astype(int)
            index_tot = np.concatenate((index_tot, index_n)).astype(int)
            index_tot = np.concatenate((index_tot, index_s)).astype(int)
            index_tot = np.unique(index_tot)[::-1] # reverse from large to small indices

            uncer_tot = np.concatenate((uncer_e, uncer_g)).astype(int)
            uncer_tot = np.concatenate((uncer_tot, uncer_n)).astype(int)
            uncer_tot = np.concatenate((uncer_tot, uncer_s)).astype(int)
            uncer_tot = np.unique(uncer_tot)

            ## record number of uncertain geometry before merging with refinement geometry
            md_uncertain.append(len(uncer_tot))

            ## filter out the unphyiscal geometries based on atom distances
            select_geom = np.array(geom)[index_tot]
            select_geom, discard_geom = self._distance_filter(allatoms, select_geom)
            select_geom = select_geom[0: self.maxsample]

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
                gap_e = np.amin(gap_e, axis=1)         # pick the smallest gap per point
                index_r = np.argsort(gap_e[self.refine_start: self.refine_end])

                refine_geom = np.array(geom)[index_r]
                refine_geom, refine_discard = self._distance_filter(allatoms, refine_geom)
                refine_geom = refine_geom[0: self.refine_num]

            else:
                index_r = []
                refine_geom = []

            ## combine select and refine geom
            select_geom += refine_geom
            
            md_nrefine.append(len(index_r))
            md_select_geom += [x.tolist() for x in select_geom]
            md_nselect.append(len(select_geom) - len(refine_geom))
            md_ndiscard.append(self.maxsample - len(select_geom) - len(refine_geom))

            ## append selected conditions
            for geo in select_geom:
                select_cond = copy.deepcopy(self.initcond[traj_id])
                select_cond.coord = geo
                self.select_cond.append(select_cond)

        ## update trajectories stat
        self.last = md_last
        self.final = md_final
        self.atoms = allatoms
        self.geom = md_geom
        self.energy = md_energy
        self.grad = md_grad
        self.nac = md_nac
        self.soc = md_soc
        self.err_e = md_err_e
        self.err_g = md_err_g
        self.err_n = md_err_n
        self.err_s = md_err_s
        self.pop = md_pop
        self.max_e = md_max_e
        self.max_g = md_max_g
        self.max_n = md_max_n
        self.max_s = md_max_s
        self.select_geom = md_select_geom
        self.nselect = md_nselect
        self.ndiscard = md_ndiscard
        self.nuncertain = md_uncertain
        self.nrefine = md_nrefine

        return self

    def _distance_filter(self, atom, geom):
        ## This function filter out unphysical geometries based on atom distances
        keep = []
        discard = []
        if len(geom) > 0:
            natom = len(geom[0])
            for geo in geom:
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
                if unphysical == 1:
                    discard.append(geo) 
                else:
                    keep.append(geo)

        return keep, discard

    def _sort_errors(self, err, threshold):
        ## This function sort the errors from the largest to the smallest
        sort_i_err = np.argsort(-err)
        sort_err  = err[sort_i_err]
        find_err  = np.argwhere(sort_err > threshold)
        index_err = sort_i_err[find_err.reshape(-1)]

        return index_err

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

    def _train_wrapper(self,fake):
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
            traj_info += '  Trajectory(ID: %6s Step: %8s State: %2s) =>\
Geometry(New: %3s Discard: %3s Refine: %s) =>\
MaxStd(Energy: %8.4f Gradient: %8.4f NAC: %8.4f SOC: %8.4f) %s\n' % (
                i + 1, 
                self.last[i], 
                self.final[i],
                self.nselect[i],
                self.ndiscard[i],
                self.nrefine[i],
                np.amax(self.err_e[i]), 
                np.amax(self.err_g[i]), 
                np.amax(self.err_n[i]), 
                np.amax(self.err_s[i]),
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
  Refinement:                 %-10s
  Discarded:                  %-10s
  Selected:                   %-10s

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
        np.sum(self.nrefine),
        np.sum(self.ndiscard),
        np.sum(self.nselect),
        self.max_e,
        self.minenergy,
        self.max_e <= self.minenergy,
        self.max_g,
        self.mingrad,
        self.max_g <= self.mingrad,
        self.max_n,
        self.minnac, 
        self.max_n <= self.minnac,
        self.max_s,
        self.minsoc,
        self.max_s <= self.minsoc)

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

        end = time.time()
        walltime = HowLong(start, end)
        tailing = 'Adaptive Sampling End: %20s Total: %20s\n' % (WhatIsTime(), walltime)
        print(tailing)
        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(tailing)
