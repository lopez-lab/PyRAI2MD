######################################################
#
# PyRAI2MD 2 module for ditributing NN training
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

import os, sys, subprocess, json, time
import numpy as np
from PyRAI2MD.Utils.timing import WhatIsTime, HowLong

class RemoteTrain:
    """ NN remote training class

        Parameters:          Type:
            keywords         dict        keyword dictionary
            id               str         training id string

        Attribute:           Type:
            keywords         dict        keyword dictionary
            title            str         calculation title
            cacldir          str         calculation directory
            pyrai2mddir      str         PyRAI2MD directory
            use_hpc          int         use HPC (1) for calculation or not(0), like SLURM.

        Functions:           Returns:
            train            dict        training metrics
    """

    def __init__(self, keywords = None, id = None):
        ## get keywords info
        self.keywords = keywords.copy()
        self.title = keywords['control']['title']
        self.pyrai2mddir = os.environ['PYRAI2MD']
        self.keywords['control']['jobtype'] = 'train'
        self.use_hpc = keywords['nn']['search']['use_hpc']
        self.retrieve = keywords['nn']['search']['retrieve']

        self.calcdir = '%s/grid-search/NN-%s-%s' % (os.getcwd(), self.title, id)

        self.runscript = """export INPUT=input.json
export WORKDIR=%s
export PYRAI2MD=%s

cd $WORKDIR
python3 $PYRAI2MD/pyrai2md.py $INPUT
""" % (     self.calcdir,
            self.pyrai2mddir)

    def _setup_hpc(self):
        ## setup HPC
        ## get python location
        pythondir = sys.executable.split('/bin')[0]

        ## read slurm template from .slurm files
        if os.path.exists('%s.slurm' % (self.title)) == True:
            with open('%s.slurm' % (self.title)) as template:
                submission = template.read()
        else:
            sys.exit('\n  FileNotFoundError\n  PyRAI2MD: looking for submission file %s.slurm' % (self.title))

        submission += self.runscript

        with open('%s/%s.sbatch' % (self.calcdir, self.title), 'w') as out:
            out.write(submission)

        return self

    def _setup_training(self):
        ## write PyRAI2MD input

        if os.path.exists(self.calcdir) == False:
            os.makedirs(self.calcdir)

        with open('%s/input.json' % (self.calcdir), 'w') as out:
            json.dump(self.keywords, out)

        ## write run script
        with open('%s/%s.sh' % (self.calcdir, self.title), 'w') as out:
            out.write(self.runscript)

        ## setup HPC setting
        if self.use_hpc == 1:
            self._setup_hpc()

        return self

    def _start_training(self):
        ## distribute NN training

        maindir = os.getcwd()
        os.chdir(self.calcdir)
        if self.use_hpc == 1:
            subprocess.run(['sbatch', '-W', '%s/%s.sbatch' % (self.calcdir, self.title)])
        else:
            subprocess.run(['bash', '%s/%s.sh' % (self.calcdir, self.title)])
        os.chdir(maindir)

        return self

    def _read_training(self):
        ## read training metrics
        if os.path.exists('%s/NN-%s.log' % (self.calcdir, self.title)) == False:
            return {'path' : self.calcdir, 'status' : 0}

        with open('%s/NN-%s.log' % (self.calcdir, self.title), 'r') as out:
            log = out.read().splitlines()

        log = log[-10:]
        for n, line in enumerate(log):
            if """&nn validation mean absolute error""" in line:
                nn1 = log[n + 4]
                nn2 = log[n + 5]
        nn1 = [float (x) for x in nn1.split()]
        nn2 = [float (x) for x in nn2.split()]

        metrics = {
            'path'   : self.calcdir,
            'status' : 1,
            'e1'     : nn1[0],
            'g1'     : nn1[1],
            'n1'     : nn1[2],
            's1'     : nn1[3],
            'e2'     : nn2[0],
            'g2'     : nn2[1],
            'n2'     : nn2[2],
            's2'     : nn2[3],
            }

        return metrics

    def train(self):
        start = time.time()

        if self.retrieve == 0:
            self._setup_training()
            self._start_training()

        metrics = self._read_training()

        end = time.time()
        walltime = HowLong(start, end)

        metrics['time'] = end - start
        metrics['walltime'] = walltime

        return metrics
