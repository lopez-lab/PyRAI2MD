######################################################
#
# PyRAI2MD 2 module for grid search
#
# Author Jingbai Li
# Sep 25 2021
#
######################################################

import os, sys, multiprocessing, time
from multiprocessing import Pool
import numpy as np

from PyRAI2MD.methods import QM
from PyRAI2MD.Machine_Learning.remote_train import RemoteTrain
from PyRAI2MD.Machine_Learning.training_data import Data
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong

class GridSearch:
    """ Grid search class

        Parameters:          Type:
            keywords         dict        keyword dictionary

        Attribute:           Type:
            layers           list        number of layers
            nodes            list        number of nodes
            batch            list        number of batch size
            l1               list        list of l1 regularization factor
            l2               list        list of l2 regularization factor
            dropout          list        list of dropout ratio
            use_hpc          int         use HPC (1) for calculation or not(0), like SLURM
            retrieve         int         retrieve training metrics


        Functions:           Returns:
            search           None        do grid-search
    """

    def __init__(self, keywords = None):
        grids      = keywords['nn']['search']

        gr_layers  = grids['depth'] 
        gr_nodes   = grids['nn_size']
        gr_batch   = grids['batch_size']
        gr_l1      = grids['reg_l1']
        gr_l2      = grids['reg_l2']
        gr_dropout = grids['dropout']

        if len(gr_layers) == 0:
            gr_layers = [1]

        if len(gr_nodes) == 0:
            gr_nodes   = [20]

        if len(gr_batch) == 0:
            gr_batch = [32]

        if len(gr_l1) == 0:
            gr_l1 = [1e-8]

        if len(gr_l2) == 0:
            gr_l2 = [1e-8]

        if len(gr_dropout) == 0:
            gr_dropout = [0.005]

        self.queue = []
        for a in gr_layers:
            for b in gr_nodes:
                for c in gr_batch:
                    for d in gr_l1:
                        for e in gr_l2:
                            for f in gr_dropout:
                                self.queue.append([a, b, c, d, e, f])

        self.keywords = keywords.copy()
        self.nsearch  = len(self.queue)
        self.version  = keywords['version']
        self.title    = keywords['control']['title']
        self.qm       = keywords['control']['qm']
        self.ml_ncpu  = keywords['control']['ml_ncpu']
        self.use_hpc  = keywords['nn']['search']['use_hpc']
        self.retrieve = keywords['nn']['search']['retrieve']
        train_data = keywords[self.qm]['train_data']
        self.data = Data()
        self.data.load(train_data)
        self.data.stat()

        title = self.keywords['nn']['train_data'].split('/')
        if len(title) == 1:
            self.keywords['nn']['train_data'] = '%s/%s' % (os.getcwd(), title[-1])

    def _update_hypers(self, keywords, hypers):
        layers, nodes, batch, l1, l2, dropout = hypers
        key = '%s_%s_%s_%s_%s_%s' % (layers, nodes, batch, l1, l2, dropout)
        variables                             = keywords.copy()
        variables['nn']['eg']['depth']        = layers
        variables['nn']['eg']['nn_size']      = nodes
        variables['nn']['eg']['batch_size']   = batch
        variables['nn']['eg']['reg_l1']       = l1
        variables['nn']['eg']['reg_l2']       = l2
        variables['nn']['eg']['dropout']      = dropout
        variables['nn']['nac']['depth']       = layers
        variables['nn']['nac']['nn_size']     = nodes
        variables['nn']['nac']['batch_size']  = batch
        variables['nn']['nac']['reg_l1']      = l1
        variables['nn']['nac']['reg_l2']      = l2
        variables['nn']['nac']['dropout']     = dropout
        variables['nn']['soc']['depth']       = layers
        variables['nn']['soc']['nn_size']     = nodes
        variables['nn']['soc']['batch_size']  = batch
        variables['nn']['soc']['reg_l1']      = l1
        variables['nn']['soc']['reg_l2']      = l2
        variables['nn']['soc']['dropout']     = dropout
        variables['nn']['eg2']['depth']       = layers
        variables['nn']['eg2']['nn_size']     = nodes
        variables['nn']['eg2']['batch_size']  = batch
        variables['nn']['eg2']['reg_l1']      = l1
        variables['nn']['eg2']['reg_l2']      = l2
        variables['nn']['eg2']['dropout']     = dropout
        variables['nn']['nac2']['depth']      = layers
        variables['nn']['nac2']['nn_size']    = nodes
        variables['nn']['nac2']['batch_size'] = batch
        variables['nn']['nac2']['reg_l1']     = l1
        variables['nn']['nac2']['reg_l2']     = l2
        variables['nn']['nac2']['dropout']    = dropout
        variables['nn']['soc2']['depth']      = layers
        variables['nn']['soc2']['nn_size']    = nodes
        variables['nn']['soc2']['batch_size'] = batch
        variables['nn']['soc2']['reg_l1']     = l1
        variables['nn']['soc2']['reg_l2']     = l2
        variables['nn']['soc2']['dropout']    = dropout

        return variables, key

    def _retrieve_data(self):
        ## retrieve training results in sequential or parallel mode
        variables_wrapper = [[n, x] for n, x in enumerate(self.queue)]

        ## adjust multiprocessing if necessary
        ncpu = 1
        if self.use_hpc > 0:
            ncpu = np.amin([self.nsearch, self.ml_ncpu])

        ## start multiprocessing
        results = [[] for x in range(self.nsearch)]
        pool = multiprocessing.Pool(processes = ncpu)
        for val in pool.imap_unordered(self._search_wrapper_hpc, variables_wrapper):
            grid_id, grid_results = val
            results[grid_id] = grid_results
        pool.close()

        return results

    def _run_search_seq(self):
        ## run training in sequential mode
        variables_wrapper = [[n, x] for n, x in enumerate(self.queue)]

        ## sequential mode
        ncpu = 1

        ## start multiprocessing
        results = [[] for x in range(self.nsearch)]
        pool = multiprocessing.Pool(processes = ncpu)
        for val in pool.imap_unordered(self._search_wrapper_seq, variables_wrapper):
            grid_id, grid_results = val
            results[grid_id] = grid_results
        pool.close()

        return results

    def _search_wrapper_seq(self, variables):
        grid_id, hypers = variables

        ## update hypers and add training data
        keywords, key = self._update_hypers(self.keywords, hypers)
        keywords[self.qm]['train_mode'] = 'training'
        keywords[self.qm]['data'] = self.data
        maindir = os.getcwd()
        calcdir = '%s/grid-search/NN-%s-%s' % (os.getcwd(), self.title, key)

        ## train on local machine
        if os.path.exists(calcdir) == False:
            os.makedirs(calcdir)

        os.chdir(calcdir)
        model = QM(self.qm, keywords = keywords)
        metrics = model.train()
        os.chdir(maindir)

        return grid_id, metrics

    def _run_search_hpc(self):
        ## wrap variables for multiprocessing
        variables_wrapper = [[n, x] for n, x in enumerate(self.queue)]

        ## adjust multiprocessing if necessary
        ncpu = np.amin([self.nsearch, self.ml_ncpu])

        ## start multiprocessing
        results = [[] for x in range(self.nsearch)]
        pool = multiprocessing.Pool(processes = ncpu)
        for val in pool.imap_unordered(self._search_wrapper_hpc, variables_wrapper):
            grid_id, grid_results = val
            results[grid_id] = grid_results
        pool.close()

        return results

    def _search_wrapper_hpc(self, variables):
        grid_id, hypers = variables

        ## update hypers
        keywords, key = self._update_hypers(self.keywords, hypers)
        keywords[self.qm]['train_mode'] = 'training'

        ## remote training in subprocess
        model = RemoteTrain(keywords = keywords, id = key)
        metrics = model.train()

        return grid_id, metrics

    def _write_summary(self, metrics):
        logpath = os.getcwd()
        summary = '  Layers   Nodes   Batch    L1        L2       Dropout    Energy1    Gradient1    NAC1        SOC1        Energy2    Gradient2    NAC2        SOC2        Time     Walltime\n'
        crashed = ''
        for n, hypers in enumerate(self.queue):

            if metrics[n]['status'] == 0:
                crashed += '%s\n' % (metrics[n]['path'])
                continue

            layers, nodes, batch, l1, l2, dropout = hypers
            e1 = metrics[n]['e1']
            g1 = metrics[n]['g1']
            n1 = metrics[n]['n1']
            s1 = metrics[n]['s1']
            e2 = metrics[n]['e2']
            g2 = metrics[n]['g2']
            n2 = metrics[n]['n2']
            s2 = metrics[n]['s2']
            t  = metrics[n]['time']
            wt = metrics[n]['walltime']
            summary += '%8d%8d%8d%10.2e%10.2e%10.2e%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%10.2e %s\n' % (
                layers,
                nodes,
                batch,
                l1,
                l2,
                dropout,
                e1,
                g1,
                n1,
                s1,
                e2,
                g2,
                n2,
                s2,
                t,
                wt)

        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(summary)

        if len(crashed) > 0:
            with open('%s/%s.crashed' % (logpath, self.title), 'w') as log:
                log.write(crashed)

        return self

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |                   Grid Search                     |
 |                                                   |
 *---------------------------------------------------*

 Number of search: %s

""" % ( self.version, self.nsearch)

        return headline

    def search(self):
        logpath = os.getcwd()
        start = time.time()
        heading = 'Grid Search Start: %20s\n%s' % (WhatIsTime(), self._heading())

        with open('%s/%s.log' % (logpath, self.title), 'w') as log:
            log.write(heading)

        if self.retrieve == 0:
            if self.use_hpc > 0:
                results = self._run_search_hpc()
            else:
                results = self._run_search_seq()
        else:
            results = self._retrieve_data()

        self._write_summary(results)
        end = time.time()
        walltime = HowLong(start, end)
        tailing = 'Grid Search End: %20s Total: %20s\n' % (WhatIsTime(), walltime)

        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(tailing)

        return self
