#####################################################
#
# PyRAI2MD 2 module for interfacing to NNsForMD(KCGNN)
#
# Author Jingbai Li
# Feb 19 2022
#
######################################################

import time, os, sys
import numpy as np
from PyRAI2MD.Machine_Learning.hypernn import SetHyperEG, SetHyperNAC, SetHyperSOC
from PyRAI2MD.Machine_Learning.permutation import PermuteMap
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong

from pyNNsMD.NNsMD import NeuralNetEnsemble

class GCNN:
    """ pyNNsMD interface

        Parameters:          Type:
            keywords         dict        keywords dict
            id               int         calculation index

        Attribute:           Type:
            hyp_eg           dict        Hyperparameters of energy gradient NN
       	    hyp_nac          dict        Hyperparameters of nonadiabatic coupling NN
       	    hyp_soc          dict     	  Hyperparameters of spin-orbit coupling NN
            x                ndarray     input structure array
            y_dict           dict        target value dict 
            pred_x           ndarray     input structure array in set prediction set
            pred_y           ndarray     target values in the prediction set

        Functions:           Returns:
            train            self        train NN for a given training set
            load             self        load trained NN for prediction
            appendix         self        fake function
            evaluate         self        run prediction

    """

    def __init__(self, keywords = None, id = None):

        set_gpu([]) #No GPU for prediction
        title           = keywords['control']['title']
        variables       = keywords['nn'].copy()
        modeldir        = variables['modeldir']
        data            = variables['data']
        nn_eg_type      = variables['nn_eg_type']
        nn_nac_type     = variables['nn_nac_type']
        nn_soc_type     = variables['nn_soc_type']
        hyp_eg          = variables['eg'].copy()
        hyp_nac         = variables['nac'].copy()
        hyp_eg2         = variables['eg2'].copy()
        hyp_nac2        = variables['nac2'].copy()
        hyp_soc         = variables['soc'].copy()
        hyp_soc2        = variables['soc2'].copy()
        eg_unit         = variables['eg_unit']
        nac_unit        = variables['nac_unit']
        soc_unit        = variables['soc_unit']
        seed            = variables['ml_seed']
        permute         = variables['permute_map']
        gpu             = variables['gpu']
        self.jobtype    = keywords['control']['jobtype']
        self.version    = keywords['version']
        self.ncpu       = keywords['control']['ml_ncpu']
        self.pred_data  = variables['pred_data']
        self.train_mode = variables['train_mode']
        self.shuffle    = variables['shuffle']
        self.natom      = data.natom
        self.nstate     = data.nstate
        self.nnac       = data.nnac
        self.nsoc       = data.nsoc

        ## set hyperparamters
        hyp_dict_eg     = SetHyperEG(hyp_eg, eg_unit, data.info)
        hyp_dict_eg2    = SetHyperEG(hyp_eg2, eg_unit, data.info)
        hyp_dict_nac    = SetHyperNAC(hyp_nac, nac_unit, data.info)
        hyp_dict_nac2   = SetHyperNAC(hyp_nac2, nac_unit, data.info)
        hyp_dict_soc    = SetHyperSOC(hyp_soc, soc_unit, data.info)
        hyp_dict_soc2   = SetHyperSOC(hyp_soc2, soc_unit, data.info)

        ## retraining has some bug at the moment, do not use
        if self.train_mode not in ['training', 'retraining', 'resample']:
            self.train_mode = 'training'
        if id == None or id == 1:
            self.name   = f"NN-{title}"
        else:
            self.name   = f"NN-{title}-{id}"
        self.silent     = variables['silent']
        self.x          = data.x
        self.pred_x     = data.pred_x
        self.pred_y     = data.pred_y

        ## convert unit of energy and force. au or si. data are in au.
        if   eg_unit == 'si':
            self.H_to_eV        = 27.211396132
            self.H_Bohr_to_eV_A = 27.211396132/0.529177249
            self.keep_eV        = 1
            self.keep_eVA       = 1
        else:
            self.H_to_eV        = 1
            self.H_Bohr_to_eV_A = 1
       	    self.keep_eV       	= 27.211396132
       	    self.keep_eVA      	= 27.211396132/0.529177249

        if   nac_unit == 'si':
            self.Bohr_to_A  = 0.529177249/27.211396132 # convert to eV/A
            self.keep_A     = 1
        elif nac_unit == 'au':
            self.Bohr_to_A  = 1                             # convert to Eh/B
            self.keep_A     = 0.529177249/27.211396132
        elif nac_unit == 'eha':
            self.Bohr_to_A  = 0.529177249                   # convert to Eh/A
            self.keep_A     = 1/27.211396132
        else:
            self.Bohr_to_A  = 1                             # convert to Eh/B
            self.keep_A     = 0.529177249/27.211396132

        ## combine y_dict
        self.y_dict = {}
        if nn_eg_type > 0:
            self.y_dict['energy_gradient'] = [data.energy * self.H_to_eV, data.grad * self.H_Bohr_to_eV_A]
        if nn_nac_type > 0:
            self.y_dict['nac'] = data.nac/self.Bohr_to_A
        if nn_soc_type > 0:
            self.y_dict['soc'] = data.soc

        ## check permuation map
        self.x, self.y_dict = PermuteMap(self.x,self.y_dict,permute,hyp_eg['val_split'])

        ## combine hypers
        self.hyper = {}
        if   nn_eg_type == 1:  # same architecture with different weight
            self.hyper['energy_gradient'] = hyp_dict_eg
        elif nn_eg_type > 1:
       	    self.hyper['energy_gradient'] = [hyp_dict_eg, hyp_dict_eg2]

        if   nn_nac_type == 1: # same architecture with different weight
       	    self.hyper['nac'] = hyp_dict_nac
       	elif nn_nac_type > 1:
            self.hyper['nac'] = [hyp_dict_nac, hyp_dict_nac2]

        if   nn_soc_type == 1: # same architecture with different weight
            self.hyper['soc'] = hyp_dict_soc
        elif nn_soc_type > 1:
            self.hyper['soc'] = [hyp_dict_soc, hyp_dict_soc2]

        ## setup GPU list
        self.gpu_list = {}
        if   gpu == 1:
            self.gpu_list['energy_gradient'] = [0, 0]
            self.gpu_list['nac'] = [0, 0]
            self.gpu_list['soc'] = [0, 0]
       	elif gpu == 2:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [0, 1]
            self.gpu_list['soc'] = [0, 1]
       	elif gpu == 3:
            self.gpu_list['energy_gradient'] = [0, 0]
            self.gpu_list['nac'] = [1, 1]
            self.gpu_list['soc'] = [2, 2]
       	elif gpu == 4:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [2, 2]
            self.gpu_list['soc'] = [3, 3]
        elif gpu == 5:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [2, 3]
            self.gpu_list['soc'] = [4, 4]
        elif gpu == 6:
            self.gpu_list['energy_gradient'] = [0, 1]
            self.gpu_list['nac'] = [2, 3]
            self.gpu_list['soc'] = [4, 5]

        ## initialize model
        if   modeldir == None or id not in [None, 1]:
            self.model = NeuralNetEnsemble(self.name)
        else:
            self.model = NeuralNetEnsemble(modeldir)

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |                  Neural Networks                  |
 |                                                   |
 *---------------------------------------------------*

 Number of atoms:  %s
 Number of state:  %s
 Number of NAC:    %s
 Number of SOC:    %s

""" % ( self.version,
        self.natom,
        self.nstate,
        self.nnac,
        self.nsoc)
 
       	return headline

    def train(self):
        start = time.time()

        self.model.create(self.hyper)

        topline = 'Neural Networks Start: %20s\n%s' % (WhatIsTime(), self._heading())
        runinfo = """\n  &nn fitting \n"""

        if self.silent == 0:
            print(topline)
            print(runinfo)

        with open('%s.log' % (self.name), 'w') as log:
            log.write(topline)
            log.write(runinfo)


        if self.train_mode == 'resample':
            out_index, out_errr, out_fiterr, out_testerr = self.model.resample(
                self.x, 
                self.y_dict,
                gpu_dist = self.gpu_list,
                proc_async = self.ncpu >= 4)
        else:
            ferr = self.model.fit(
                self.x,
                self.y_dict,
                gpu_dist = self.gpu_list,
                proc_async = self.ncpu >= 4,
                fitmode = self.train_mode,
                random_shuffle = self.shuffle)

            #self.model.save()
            err_e1 = 0
            err_e2 = 0
            err_g1 = 0
            err_g2 = 0
            err_n1 = 0
            err_n2 = 0
            err_s1 = 0
            err_s2 = 0
            if 'energy_gradient' in ferr.keys():
                err_e1 = ferr['energy_gradient'][0][0]
                err_e2 = ferr['energy_gradient'][1][0]
                err_g1 = ferr['energy_gradient'][0][1]
       	       	err_g2 = ferr['energy_gradient'][1][1]

            if 'nac' in ferr.keys():
                err_n1 = ferr['nac'][0]
                err_n2 = ferr['nac'][1]

            if 'soc' in ferr.keys():
                err_s1 = ferr['soc'][0]
                err_s2 = ferr['soc'][1]

            metrics = {
                'e1' : err_e1 * self.keep_eV,
                'g1' : err_g1 * self.keep_eVA,
                'n1' : err_n1 / self.keep_A,
                's1' : err_s1,
                'e2' : err_e2 * self.keep_eV,
                'g2' : err_g2 * self.keep_eVA,
                'n2' : err_n2 / self.keep_A,
                's2' : err_s2}

            train_info="""
  &nn validation mean absolute error
-------------------------------------------------------
      energy       gradient       nac          soc
        eV           eV/A         eV/A         cm-1
  %12.8f %12.8f %12.8f %12.8f
  %12.8f %12.8f %12.8f %12.8f

""" % (metrics['e1'], metrics['g1'], metrics['n1'], metrics['s1'],
       metrics['e2'], metrics['g2'], metrics['n2'], metrics['s2'])

        end = time.time()
        walltime = HowLong(start,end)
        endline = 'Neural Networks End: %20s Total: %20s\n' % (WhatIsTime(), walltime)

        if self.silent == 0:
            print(train_info)
            print(endline)

        with open('%s.log' % (self.name), 'a') as log:
            log.write(train_info)
            log.write(endline)

        metrics['time'] = end - start
        metrics['walltime'] = walltime
        metrics['path'] = os.getcwd()
        metrics['status'] = 1

        return metrics

    def load(self):
        self.model.load()

        return self

    def	appendix(self,addons):
       	## fake	function does nothing

       	return self

    def _qm(self, traj):
        ## run psnnsmd for QM calculation

        xyz = traj.coord.reshape((1, self.natom, 3))
        y_pred,y_std=self.model.call(xyz)

        ## initialize return values
        energy = []
        gradient = []
        nac = []
        soc = []
        err_e = 0
       	err_g =	0
       	err_n =	0
       	err_s =	0

        ## update return values
        if 'energy_gradient' in y_pred.keys():
            e_pred = y_pred['energy_gradient'][0] / self.H_to_eV
            g_pred = y_pred['energy_gradient'][1] / self.H_Bohr_to_eV_A
            e_std = y_std['energy_gradient'][0] / self.H_to_eV
            g_std = y_std['energy_gradient'][1] / self.H_Bohr_to_eV_A
            energy = e_pred[0]
            gradient = g_pred[0]
            err_e = np.amax(e_std)
            err_g = np.amax(g_std)

        if 'nac' in y_pred.keys():
            n_pred = y_pred['nac']*self.Bohr_to_A
            n_std = y_std['nac']*self.Bohr_to_A
            nac = n_pred[0]
            err_n = np.amax(n_std)

        if 'soc' in y_pred.keys():
            s_pred = y_pred['soc']
            s_std = y_std['soc']
            soc = s_pred[0]
            err_s = np.amax(s_std)

        return energy, gradient, nac, soc, err_e, err_g, err_n, err_s

    def _predict(self, x):
        ## run psnnsmd for model testing

        batch = len(x)

        y_pred, y_std = self.model.predict(x)

        ## load values from prediction set
        pred_e = self.pred_y['energy']
        pred_g = self.pred_y['grad']
        pred_n = self.pred_y['nac']
        pred_s = self.pred_y['soc']

        ## initialize errors
        de_max = np.zeros(batch)
        dg_max = np.zeros(batch)
        dn_max = np.zeros(batch)
        ds_max = np.zeros(batch)

        ## update errors
        if 'energy_gradient' in y_pred.keys():
            e_pred = y_pred['energy_gradient'][0] / self.H_to_eV
            g_pred = y_pred['energy_gradient'][1] / self.H_Bohr_to_eV_A
            e_std = y_std['energy_gradient'][0] / self.H_to_eV 
            g_std = y_std['energy_gradient'][1] / self.H_Bohr_to_eV_A
            de = np.abs(pred_e - e_pred)
            dg = np.abs(pred_g - g_pred)
            de_max = np.amax(de.reshape((batch, -1)), axis = 1)
            dg_max = np.amax(dg.reshape((batch, -1)), axis = 1)

            val_out = np.concatenate((pred_e.reshape((batch, -1)), e_pred.reshape((batch, -1))), axis = 1)
            std_out = np.concatenate((de.reshape((batch, -1)), e_std.reshape((batch, -1))), axis = 1)
            np.savetxt('%s-e.pred.txt' % (self.name), np.concatenate((val_out, std_out), axis = 1))

            val_out = np.concatenate((pred_g.reshape((batch, -1)), g_pred.reshape((batch, -1))), axis = 1)
       	    std_out = np.concatenate((dg.reshape((batch, -1)), g_std.reshape((batch, -1))), axis = 1)
            np.savetxt('%s-g.pred.txt' % (self.name), np.concatenate((val_out, std_out), axis = 1))

        if 'nac' in y_pred.keys():
            n_pred = y_pred['nac'] * self.Bohr_to_A
            n_std = y_std['nac'] * self.Bohr_to_A
            dn = np.abs(pred_n - n_pred)
            dn_max = np.amax(dn.reshape((batch, -1)), axis = 1)

            val_out = np.concatenate((pred_n.reshape((batch, -1)), n_pred.reshape((batch, -1))), axis = 1)
       	    std_out = np.concatenate((dn.reshape((batch, -1)), n_std.reshape((batch, -1))), axis = 1)
            np.savetxt('%s-n.pred.txt' % (self.name), np.concatenate((val_out, std_out), axis = 1))


        if 'soc' in y_pred.keys():
            s_pred = y_pred['soc']
            s_std = y_std['soc']
       	    ds = np.abs(pred_s - s_pred)
            ds_max = np.amax(ds.reshape((batch, -1)), axis = 1)

            val_out = np.concatenate((pred_s.reshape((batch, -1)), s_pred.reshape((batch, -1))), axis = 1)
       	    std_out = np.concatenate((ds.reshape((batch, -1)), s_std.reshape((batch, -1))), axis = 1)
            np.savetxt('%s-s.pred.txt' % (self.name), np.concatenate((val_out, std_out), axis = 1))

        output = ''
        for i in range(batch):
            output += '%5s %8.4f %8.4f %8.4f %8.4f\n' % (i + 1, de_max[i], dg_max[i], dn_max[i], ds_max[i])

        with open('max_abs_dev.txt', 'w') as out:
            out.write(output)

        return self

    def evaluate(self, traj):
        ## main function to run pyNNsMD and communicate with other PyRAIMD modules

        if   self.jobtype == 'prediction' or self.jobtype == 'predict':
            self._predict(self.pred_x)
        else:
            energy, gradient, nac, soc, err_energy, err_grad, err_nac, err_soc = self._qm(traj)
            traj.energy = np.copy(energy)
            traj.grad = np.copy(gradient)
            traj.nac = np.copy(nac)
            traj.soc = np.copy(soc)
            traj.err_energy = err_energy
            traj.err_grad = err_grad
            traj.err_nac = err_nac
            traj.err_soc = err_soc
            traj.status = 1

            return traj
