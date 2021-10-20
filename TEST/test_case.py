######################################################
#
# PyRAI2MD test cases
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

test_first_run         = 1
test_bagel             = 1
test_molcas            = 1
test_molcas_tinker     = 1
test_fssh              = 1
testh_gsh              = 1
test_nn                = 1
test_grid_search       = 1
test_aimd              = 1
test_mixaimd           = 1
test_adaptive_sampling = 1

import time, datetime, os

class TestCase:
    """ PyRAI2MD test cases
    1. check code completeness
        first_run

    2. test qc methods
        bagel local hpc
        molcas local hpc
        molcas_tinker local hpc

    3. test ml method
        train and prediction
        grid_search seq, hpc

    4. test surface hopping
        fssh
        gsh nac soc

    5. test md
        aimd
        mixaimd
        ensemble

    6. test adaptive sampling
        adaptive sampling

    7. test utils
        alignment
        coordinates
        sampling

"""
    def __init__(self, logo):
        self.logo = logo
        self.register = {
            'first_run'          : test_first_run,
            'bagel'              : test_bagel,
            'molcas'             : test_molcas,
            'molcas_tinker'      : test_molcas_tinker,
            'fssh'               : test_fssh,
            'gsh'                : testh_gsh,
            'neural_network'     : test_nn,
            'grid_search'        : test_grid_search,
            'aimd'               : test_aimd,
            'mixaimd'            : test_mixaimd,
            'adaptive_sampling'  : test_adaptive_sampling,
            }

        self.test_func = {}
        testdir = os.environ['PYRAI2MD']
        if os.path.exists('%s/TEST/first_run/test_first_run.py' % (testdir)) == True:
            from TEST.first_run.test_first_run import FirstRun
            self.test_func['first_run'] = FirstRun 

        if os.path.exists('%s/TEST/bagel/test_bagel.py' % (testdir)) == True:
            from TEST.bagel.test_bagel import TestBagel
            self.test_func['bagel'] = TestBagel

        if os.path.exists('%s/TEST/molcas/test_molcas.py' % (testdir)) == True:
            from TEST.molcas.test_molcas import TestMolcas
            self.test_func['molcas'] = TestMolcas 

        if os.path.exists('%s/TEST/molcas_tinker/test_molcas_tinker.py' % (testdir)) == True:
            from TEST.molcas_tinker.test_molcas_tinker import TestMolcasTinker
            self.test_func['molcas_tinker'] = TestMolcasTinker

        if os.path.exists('%s/TEST/neural_network/test_nn.py' % (testdir)) == True:
            from TEST.neural_network.test_nn import TestNN
            self.test_func['neural_network'] = TestNN

        if os.path.exists('%s/TEST/grid_search/test_grid_search.py' % (testdir)) == True:
            from TEST.grid_search.test_grid_search import TestGridSearch
            self.test_func['grid_search'] = TestGridSearch

        if os.path.exists('%s/TEST/fssh/test_fssh.py' % (testdir)) == True:
            from TEST.fssh.test_fssh import TestFSSH
            self.test_func['fssh'] = TestFSSH

        if os.path.exists('%s/TEST/gsh/test_gsh.py' % (testdir)) == True:
            from TEST.gsh.test_gsh import TestGSH
            self.test_func['gsh'] = TestGSH

        if os.path.exists('%s/TEST/aimd/test_aimd.py' % (testdir)) == True:
            from TEST.aimd.test_aimd import TestAIMD
            self.test_func['aimd'] = TestAIMD

        if os.path.exists('%s/TEST/mixaimd/test_mixaimd.py' % (testdir)) == True:
            from TEST.mixaimd.test_mixaimd import TestMIXAIMD
            self.test_func['mixaimd'] = TestMIXAIMD

        if os.path.exists('%s/TEST/adaptive_sampling/test_adaptive_sampling.py' % (testdir)) == True:
            from TEST.adaptive_sampling.test_adaptive_sampling import TestAdaptiveSampling
            self.test_func['adaptive_sampling'] = TestAdaptiveSampling

    def run(self):
        heading = '''
%s

-------------------------------------------------------
                       PyRAI2MD
               _____  ____  ____  _____   
                 |    |___  |___`   |
                 |    |___  .___|   |
-------------------------------------------------------

''' % (self.logo)
        with open('test.log', 'w') as out:
            out.write(heading)

        print(heading)
        ntest = len(self.register)
        n = 0
        for testcase, status in self.register.items():
            n += 1
            code = 'PASSED'
            print('TEST %3s of %3s: %-20s ...      ' % (n, ntest, testcase), end = '')
            summary = ''
            if status == 0:
                summary += '\nTEST %-20s Skipped\n\n' % (testcase) 
                code = 'SKIPPED'
            else:
                start = time.time()

                summary += '==TEST==> %-20s Start:  %s\n' % (testcase, WhatIsTime())
                results, code = self.test_func[testcase]()
                summary += results
                summary += '==TEST==> %-20s End:    %s\n' % (testcase, WhatIsTime())

                end = time.time()
                walltime = HowLong(start, end)
                summary += '==TEST==> %-20sUsed:    %s\n\n' % (testcase, walltime)
            print(code)

            with open('test.log', 'a') as out:
                out.write(summary)


def WhatIsTime():
    ## This function return current time

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def HowLong(start, end):
    ## This function calculate time between start and end

    walltime = end-start
    walltime = '%5d days %5d hours %5d minutes %5d seconds' % (
        int(walltime / 86400),
        int((walltime % 86400) / 3600),
        int(((walltime % 86400) % 3600) / 60),
        int(((walltime % 86400) % 3600) % 60))
    return walltime

