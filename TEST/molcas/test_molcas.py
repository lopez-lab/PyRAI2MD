######################################################
#
# PyRAI2MD test Molcas
#
# Author Jingbai Li
# Sep 30 2021
#
######################################################

import os, sys, shutil, json, subprocess

def TestMolcas():
    """ molcas test

    1. CASSCF orbital
    2. CASSCF energy, gradient, NAC and SOC

    """
    pyrai2mddir = os.environ['PYRAI2MD']
    testdir = '%s/molcas' % (os.getcwd())
    record = {
        'casscf'   : 'FileNotFound',
        'coupling' : 'FileNotFound',
        'coord'    : 'FileNotFound',
        'MOLCAS'   : 'VariableNotFound',
        }

    casscfpath = '%s/TEST/molcas/molcas_data/c2h4-casscf.inp' % (pyrai2mddir)
    if os.path.exists(casscfpath):
        record['casscf'] = casscfpath

    couplingpath = '%s/TEST/molcas/molcas_data/c2h4.inp' % (pyrai2mddir)
    if os.path.exists(couplingpath):
        record['coupling'] = couplingpath

    coordpath = '%s/TEST/molcas/molcas_data/c2h4.xyz' % (pyrai2mddir)
    if os.path.exists(coordpath):
       	record['coord']	= coordpath

    if 'MOLCAS' in os.environ:
        record['MOLCAS'] = os.environ['MOLCAS']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |            MOLCAS Test Calculation                |
 |                                                   |
 *---------------------------------------------------*

 Some enviroment variables are needed for this test:

    export MOLCAS=/path

 Check files and settings:
-------------------------------------------------------
"""
    for key, location in record.items():
        summary += ' %-10s %s\n' % (key, location)

    for key, location in record.items():
        if location == 'FileNotFound':
            summary += '\n Test files are incomplete, please download it again, skip test\n\n'
            return summary, 'FAILED(test file unavailable)'
        if location == 'VariableNotFound':
            summary += '\n Environment variables are not set, cannot find program, skip test\n\n'
            return summary, 'FAILED(enviroment variable missing)'

    CopyInput(record, testdir)
    Setup(record, testdir)

    summary += """
 Copy files:
 %-10s --> %s/c2h4-casscf.xyz
 %-10s --> %s/c2h4.inp (renamed to c2h4.molcas)
 %-10s --> %s/c2h4.xyz

 Run MOLCAS CASSCF:
""" % ('casscf', testdir, 'coupling', testdir, 'coord', testdir)

    code = RunCASSCF(record, testdir)

    if code == 'PASSED':
        summary += """
 CASSCF done, entering state coupling calculation

"""
    else:
        summary += """
 CASSCF failed, stop here
"""
        return summary, code

    results, code = RunCASPT2(record, testdir, pyrai2mddir)
   
    summary += """
-------------------------------------------------------
                     MOLCAS OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % (results)   
    return summary, code
 
def CopyInput(record, testdir):
    if os.path.exists(testdir) == False:
        os.makedirs(testdir)

    shutil.copy2(record['casscf'], '%s/c2h4-casscf.inp' % (testdir))
    shutil.copy2(record['coupling'], '%s/c2h4.molcas' % (testdir))
    shutil.copy2(record['coord'], '%s/c2h4.xyz' % (testdir))

def Setup(record, testdir):
    input = """&CONTROL
title         c2h4
qc_ncpu       1
jobtype       sp
qm            molcas

&Molcas
molcas         %s
molcas_calcdir %s

&Molecule
ci       3 2
spin     0 1
coupling 1 2, 2 3, 4 5, 2 4, 2 5 
""" % ( record['MOLCAS'],
        testdir)

    runscript = """
export INPUT=c2h4-casscf
export MOLCAS=%s
export MOLCAS_PROJECT=$INPUT
export MOLCAS_WORKDIR=$PWD

$MOLCAS/bin/pymolcas -f $INPUT.inp -b 1
rm -r $MOLCAS_PROJECT
""" %  (record['MOLCAS'])

    with open('%s/test_inp' % (testdir), 'w') as out:
        out.write(input)

    with open('%s/molcascasscf.sh' % (testdir), 'w') as out:
        out.write(runscript)

def Collect(testdir):
    with open('%s/c2h4.log' % (testdir), 'r') as logfile:
        log = logfile.read().splitlines()
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def RunCASSCF(record, testdir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('bash molcascasscf.sh', shell = True)
    os.chdir(maindir)    
    with open('%s/c2h4-casscf.log' % (testdir), 'r') as logfile:
        log = logfile.read().splitlines()
    code = 'FAILED(casscf runtime error)'
    for line in log[-10:]:
        if 'Happy' in line:
            code = 'PASSED'
            shutil.copy2('%s/c2h4-casscf.RasOrb' % (testdir), '%s/c2h4.StrOrb' % (testdir))
    return code

def RunCASPT2(record, testdir, pyrai2mddir):
    maindir = os.getcwd()
    os.chdir(testdir)
    subprocess.run('python3 %s/pyrai2md.py test_inp' % (pyrai2mddir), shell = True)
    os.chdir(maindir)
    results = Collect(testdir)
    if len(results.splitlines()) < 13:
        code = 'FAILED(coupling runtime error)'
    else:
        code = 'PASSED'
    return results, code
