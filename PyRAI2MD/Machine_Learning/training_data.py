#####################################################
#
# PyRAI2MD 2 module for processing training data
#
# Author Jingbai Li
# Sep 22 2021
#
######################################################

import os, sys, json
import numpy as np

class Data:
    """ Training data class

        Parameters:          Type:
            None

        Attribute:           Type:       
            natom            int         number of atoms
            nstate           int         number of states
            nnac             int         number of nonadiabatic couplings
       	    nsoc             int    	 number of spin-orbit couplings
       	    info             dict        data size info dict
            xyz              ndarray     coordinates array
            energy           ndarray     energy array
            grad             ndarray     gradient array
            nac              ndarray     nonadiabatic coupling array
            soc              ndarray     spin-orbit coulping array
            x                ndarray     training set coordinates
            pred_x           ndarray     prediction set coordinates
            pred_y           dict        prediction set target values
            max_xx           float       maximum value
            min_xx           float       minimum value
            mid_xx           float       middle value
            dev_xx           float       deviation value
            avg_xx           float       mean value
            std_xx           float       standard deviation

        Functions:           Returns:
            load             self        load data
            append           self        add new data
            save             self        save data
            stat             self        update data statisitcs (max, min, mid, dev, mean, std)
    """

    def __init__(self):
        
        self.natom      = 0
        self.nstate     = 0
        self.nnac       = 0
        self.nsoc       = 0
        self.info       = {}
        self.xyz        = np.zeros(0)
        self.energy     = np.zeros(0)
        self.grad       = np.zeros(0)
        self.nac        = np.zeros(0)
        self.soc        = np.zeros(0)
        self.x          = np.zeros(0)
        self.pred_x     = np.zeros(0)
        self.pred_y     = {}
        self.max_energy = 0
        self.max_grad   = 0
       	self.max_nac    = 0
       	self.max_soc    = 0
       	self.min_energy	= 0
       	self.min_grad   = 0
        self.min_nac    = 0
        self.min_soc    = 0
       	self.mid_energy	= 0
       	self.mid_grad   = 0
        self.mid_nac    = 0
        self.mid_soc    = 0
       	self.dev_energy	= 0
       	self.dev_grad   = 0
        self.dev_nac    = 0
        self.dev_soc    = 0
       	self.avg_energy	= 0
       	self.avg_grad   = 0
        self.avg_nac    = 0
        self.avg_soc    = 0
       	self.std_energy	= 0
       	self.std_grad   = 0
        self.std_nac    = 0
        self.std_soc    = 0

    def _load_training_data (self, file):
        with open('%s' % file,'r') as indata:
            data=json.load(indata)

        if   isinstance(data, list): ## old format
            natom, nstate, xyz, invr, energy, grad, nac, ci, mo = data
            self.natom  = int(natom)
            self.nstate = int(nstate)
            self.nnac   = int(nstate * (nstate - 1) / 2)
       	    self.xyz    = np.array(xyz)
       	    self.energy	= np.array(energy)
       	    self.grad  	= np.array(grad)
            self.nac    = np.array(nac)

        elif isinstance(data, dict): ## new format
            self.natom  = int(data['natom'])
            self.nstate = int(data['nstate'])
            self.nnac	= int(data['nnac'])
            self.nsoc	= int(data['nsoc'])
            self.xyz    = np.array(data['xyz'])
            self.energy = np.array(data['energy'])
            self.grad   = np.array(data['grad'])
            self.nac    = np.array(data['nac'])
            self.soc    = np.array(data['soc']) 

        else:
            sys.exit('\n  FileTypeError\n  PyRAI2MD: cannot recognize training data format %s' % (file))

        self.x = np.array(self.xyz[:, :, 1: 4]).astype(float)
        self.info = {
            'natom'  : self.natom,
            'nstate' : self.nstate,
            'nnac'   : self.nnac,
            'nsoc'   : self.nsoc,
            }

        return self

    def _load_prediction_data(self, file):
        with open('%s' % (file), 'r') as indata:
            data=json.load(indata)

        if   isinstance(data, list): ## old format
            natom, nstate, xyz, invr, energy, grad, nac, ci, mo = data
            self.pred_x = np.array(xyz)[:, :, 1: 4].astype(float)
            self.pred_y = {
                'energy' : np.array(energy),
                'grad'   : np.array(grad),
                'nac'    : np.array(nac),
                'soc'    : 0
                }

        elif isinstance(data, dict): ## new format
            self.pred_x = np.array(data['xyz'])[:, :, 1: 4].astype(float)
            self.pred_y = {
       	       	'energy' : np.array(data['energy']),
       	       	'grad' 	 : np.array(data['grad']),
       	       	'nac'  	 : np.array(data['nac']),
       	       	'soc'  	 : np.array(data['soc']),         
                }

        else:
            sys.exit('\n  FileTypeError\n  PyRAI2MD: cannot recognize prediction data format %s' % (file))

        return self

    def load(self, file, filetype = 'train'):
        if os.path.exists(file) == False:
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for training data  %s for %s' % (file, filetype))

        if   filetype == 'train':
            self._load_training_data(file)
        elif filetype == 'prediction':
            self._load_prediction_data(file)

        return self

    def save(self, file):
        batch = len(self.xyz)
        data = {
            'natom'  : self.natom,
            'nstate' : self.nstate,
            'nnac'   : self.nnac,
            'nsoc'   : self.nsoc,
            'xyz'    : self.xyz.tolist(),
            'energy' : self.energy.tolist(),
            'grad'   : self.grad.tolist(),
            'nac'    : self.nac.tolist(),
            'soc'    : self.soc.tolist(),
            }

        with open('New-data%s-%s.json' % (batch, file), 'w') as outdata:
            json.dump(data,outdata)


        return self

    def append(self, newdata):
        new_xyz, new_energy, new_grad, new_nac, new_soc = newdata
        self.xyz    = np.concatenate((self.xyz, new_xyz))
        self.energy = np.concatenate((self.energy, new_energy))
        self.grad   = np.concatenate((self.grad, new_grad))
        self.nac    = np.concatenate((self.nac, new_nac))
        self.soc    = np.concatenate((self.soc, new_soc))
        self.x      = np.array(self.xyz[:, :, 1: 4]).astype(float)

        return self

    def stat(self):
        if len(self.energy[0]) > 0:
            self.max_energy = np.amax(self.energy)
            self.min_energy = np.amin(self.energy)
            self.mid_energy = (self.max_energy + self.min_energy) / 2
            self.dev_energy = (self.max_energy - self.min_energy) / 2
            self.avg_energy = np.mean(self.energy)
            self.std_energy = np.std(self.energy)

        if len(self.grad[0]) > 0:
            self.max_grad   = np.amax(self.grad)
            self.min_grad   = np.amin(self.grad)
            self.mid_grad   = (self.max_grad + self.min_grad) / 2
            self.dev_grad   = (self.max_grad - self.min_grad) / 2
            self.avg_grad   = np.mean(self.grad)
            self.std_grad   = np.std(self.grad)

        if len(self.nac[0]) > 0:
            self.max_nac    = np.amax(self.nac)
            self.min_nac    = np.amin(self.nac)
            self.mid_nac    = (self.max_nac + self.min_nac) / 2
            self.dev_nac    = (self.max_nac - self.min_nac) / 2
            self.avg_nac    = np.mean(self.nac)
            self.std_nac    = np.std(self.nac)

        if len(self.soc[0]) > 0:
            self.max_soc    = np.amax(self.soc)
            self.min_soc    = np.amin(self.soc)
            self.mid_soc    = (self.max_soc + self.min_soc) / 2
            self.dev_soc    = (self.max_soc - self.min_soc) / 2
            self.avg_soc    = np.mean(self.soc)
            self.std_soc    = np.std(self.soc)

        return self
