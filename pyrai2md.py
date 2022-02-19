######################################################
#
# PyRAI2MD 2 main function
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os, sys, json

if 'PYRAI2MD' in os.environ.keys():
    sys.path.append('%s/PyRAI2MD/Machine_Learning' % os.environ['PYRAI2MD'])
else:
    sys.exit("""
  Environment variable "PYRAI2MD" is not set
  Please add the following command in your script

     export PYRAI2MD=/path/to/PyRAI2MD
""")

if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from PyRAI2MD.variables import ReadInput, StartInfo
from PyRAI2MD.methods import QM
from PyRAI2MD.Molecule.trajectory import Trajectory
from PyRAI2MD.Dynamics.aimd import AIMD
from PyRAI2MD.Dynamics.mixaimd import MIXAIMD
from PyRAI2MD.Dynamics.single_point import SinglePoint
from PyRAI2MD.Dynamics.hop_probability import HopProb
from PyRAI2MD.Machine_Learning.training_data import Data
from PyRAI2MD.Machine_Learning.grid_search import GridSearch
from PyRAI2MD.Machine_Learning.adaptive_sampling import AdaptiveSampling
from PyRAI2MD.Utils.coordinates import PrintCoord
from PyRAI2MD.Utils.sampling import Sampling
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong
from PyRAI2MD.Utils.logo import Logo

class PYRAI2MD:
    """ Main PyRAI2MD interface

        Parameters:          Type:
            input            str         input file name

        Attribute:           Type:
            keywords         dict        keyword dictionary
            
        Functions:           Returns:
            run              None        run PyRAI2MD calculation
            test             None        run PyRAI2MD testcases
    """

    def __init__(self, input):
        ## check input
        version='2.1 alpha'
        self.logo = Logo(version)
        if input == None: print(self.logo)

        if input != 'quicktest' and input != None:
            ## read input
            if os.path.exists(input) == False:
                sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for input file %s' % (input))
            input_dict = self._load_input(input)
            self.keywords = ReadInput(input_dict)
            input_info = StartInfo(self.keywords)

            ## get control info
            self.keywords['version'] = self._version_info(version, input_info)
            self.title   = self.keywords['control']['title']
            self.jobtype = self.keywords['control']['jobtype']
            self.qm      = self.keywords['control']['qm']
            self.abinit  = self.keywords['control']['abinit']

    def _version_info(self, version, input_info):
        info="""%s

%s

""" % (Logo(version), input_info)

        return info

    def _load_input(self, input):
        with open(input, 'r') as file:
            try:
                input_dict = json.load(file)

            except:
                with open(input, 'r') as file:
                    input_dict = file.read().split('&')

        return input_dict

    def _machine_learning(self):
        train_data = self.keywords[self.qm]['train_data']
        pred_data = self.keywords[self.qm]['pred_data']
        data = Data()

        if   self.jobtype == 'train':            
            ## get training data
            data.load(train_data)
            data.stat()
            self.keywords[self.qm]['data'] = data

            ## create model
            model = QM(self.qm, keywords = self.keywords, id = None)
            model.train()

        elif self.jobtype == 'prediction' or self.jobtype == 'predict':
            ## get training data and prediction data
            data.load(train_data)
            data.load(pred_data, filetype = 'prediction')
            data.stat()
            self.keywords[self.qm]['data'] = data

            ## create model
            model = QM(self.qm, keywords = self.keywords, id = None)
            model.load()
            model.evaluate(None)

        return self

    def _single_point(self):
        ## create a trajectory and method model
        traj = Trajectory(self.title, keywords = self.keywords)
        method = QM(self.qm, keywords = self.keywords, id=None)
        method.load()

        sp = SinglePoint(trajectory = traj,
                         keywords = self.keywords,
                         qm = method,
                         id = None,
                         dir = None)
        sp.run()
        return self

    def _hop_probability(self):
        ## create a trajectory and method model
        traj = Trajectory(self.title, keywords = self.keywords)
        hop = HopProb(trajectory = traj, keywords = self.keywords)
        hop.run()
        return self

    def _dynamics(self):
        ## get md info
        md        = self.keywords['md']
        initcond  = md['initcond']
        ninitcond = md['ninitcond']
        method    = md['method']
        format    = md['format']
        gl_seed   = md['gl_seed']
        temp      = md['temp']

        ## get molecule info
        if initcond == 0:
            mol = self.title
        else:
            ## use sampling method to generate intial condition
            mol = Sampling(self.title, ninitcond, gl_seed, temp, method, format)[-1]
            ## save sampled geometry and velocity
            xyz, velo = ReadInitcond(mol)
            initxyz_info = '%d\n%s\n%s' % (
                len(xyz),
                '%s sampled geom %s at %s K' % (method, ninitcond, temp),
                PrintCoord(xyz))

            with open('%s.xyz' % (self.title), 'w') as initxyz:
                initxyz.write(initxyz_info)

            with open('%s.velo' % (self.title), 'w') as initvelo:
                np.savetxt(initvelo, velo, fmt='%30s%30s%30s')

        ## create a trajectory and method model
        if self.qm == 'nn':
            train_data = self.keywords[self.qm]['train_data']
            data = Data()
            data.load(train_data)
            data.stat()
            self.keywords[self.qm]['data'] = data

        traj = Trajectory(mol, keywords = self.keywords)
        method = QM(self.qm, keywords = self.keywords, id=None)
        method.load()
        aimd = AIMD(trajectory = traj,
                    keywords = self.keywords,
                    qm = method,
                    id = None,
                    dir = None)
        aimd.run()

        return self

    def _hybrid_dynamics(self):
        ## get md info
        md        = self.keywords['md']
        initcond  = md['initcond']
        ninitcond = md['ninitcond']
        method    = md['method']
        format    = md['format']
        gl_seed   = md['gl_seed']
        temp      = md['temp']

        ## get molecule info
        if initcond == 0:
            mol = self.title
        else:
            ## use sampling method to generate intial condition
            mol = Sampling(self.title, ninitcond, gl_seed, temp, method, format)[-1]
            ## save sampled geometry and velocity
            xyz, velo = ReadInitcond(mol)
            initxyz_info = '%d\n%s\n%s' % (
                len(xyz),
                '%s sampled geom %s at %s K' % (method, ninitcond, temp),
                PrintCoord(xyz))

            with open('%s.xyz' % (title), 'w') as initxyz:
                initxyz.write(initxyz_info)

            with open('%s.velo' % (title), 'w') as initvelo:
                np.savetxt(initvelo, velo, fmt='%30s%30s%30s')

        ## create a trajectory and method model
        traj = Trajectory(mol, keywords = self.keywords)
        ref = QM(self.abinit, keywords = self.keywords, id = None)
        ref.load()

        train_data = self.keywords[self.qm]['train_data']
        data = Data()
        data.load(train_data)
        data.stat()
        self.keywords[self.qm]['data'] = data
        method = QM(self.qm, keywords = self.keywords, id=None)
        method.load()

        mixaimd = MIXAIMD(trajectory = traj,
                          keywords = self.keywords,
                          qm = method,
                          ref = ref,
                          id = None,
                          dir = None)
        mixaimd.run()

        return self

    def	_active_search(self):
        sampling = AdaptiveSampling(keywords = self.keywords)
        sampling.search()

        return self

    def _grid_search(self):
        grid = GridSearch(keywords = self.keywords)
        grid.search()

        return self

    def run(self):
        job_func={
        'sp'         : self._single_point,
        'md'         : self._dynamics,
        'hop'        : self._hop_probability,
        'hybrid'     : self._hybrid_dynamics,
        'adaptive'   : self._active_search,
        'train'      : self._machine_learning,
        'prediction' : self._machine_learning,
        'predict'    : self._machine_learning,
        'search'     : self._grid_search,
        }

        job_func[self.jobtype]()

    def test(self):
        testdir = os.environ['PYRAI2MD']
        if os.path.exists('%s/TEST' % (testdir)) == False:
            print(self.logo)
            print('\n PyRAIMD2 has no testcases to run, good luck! \n')
            exit()
        else:
            from TEST.test_case import TestCase
            test = TestCase(self.logo)
            test.run()

if __name__ == '__main__':
    pmd = PYRAI2MD
    if len(sys.argv) < 2:
        pmd(None)
        sys.exit("""
  PyRAI2MD input file is not set
  Please add the following command in your script

     export PYRAI2MD=/path/to/PyRAI2MD
     python3 $PYRAI2MD/pyrai2md.py input
""")
    else:
        if sys.argv[1] == 'quicktest':
            pmd(sys.argv[1]).test()
        else:
            pmd(sys.argv[1]).run()
