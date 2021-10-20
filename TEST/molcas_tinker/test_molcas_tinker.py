######################################################
#
# PyRAI2MD test Molcas/Tinker
#
# Author Jingbai Li
# Sep 30 2021
#
######################################################

import os, sys, shutil, json, subprocess

def TestMolcasTinker():
    """ molcas/tinker test

    1. Reading external tinker xyz
    2. QMMM CASSCF energy, gradient, NAC and SOC

    """

    pyrai2mddir = os.environ['PYRAI2MD']
    testdir = '%s/molcas_tinker' % (os.getcwd())
    record = {
        'energy'   : 'FileNotFound',
        'coupling' : 'FileNotFound',
        'orbital'  : 'FileNotFound',
        'coord'    : 'FileNotFound',
        'velo'     : 'FileNotFound',
        'qmmm_key' : 'FileNotFound',
        'qmmm_xyz' : 'FileNotFound',
        'qmmm_prm' : 'FileNotFound',
        'MOLCAS'   : 'VariableNotFound',
        'TINKER'   : 'VariableNotFound',
        }


    filepath = '%s/TEST/molcas_tinker/qmmm_data/PSB5-sp.inp' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['energy'] = filepath

    couplingpath = '%s/TEST/molcas_tinker/qmmm_data/PSB5.inp' % (pyrai2mddir)
    if os.path.exists(couplingpath):
        record['coupling'] = couplingpath

    coordpath = '%s/TEST/molcas_tinker/qmmm_data/PSB5.xyz' % (pyrai2mddir)
    if os.path.exists(coordpath):
       	record['coord']	= coordpath

    filepath = '%s/TEST/molcas_tinker/qmmm_data/PSB5.txyz' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_xyz'] = filepath

    filepath = '%s/TEST/molcas_tinker/qmmm_data/PSB5.velocity.xyz' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['velo'] = filepath

    filepath = '%s/TEST/molcas_tinker/qmmm_data/PSB5.StrOrb' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['orbital'] = filepath

    filepath = '%s/TEST/molcas_tinker/qmmm_data/PSB5.key' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_key'] = filepath

    filepath = '%s/TEST/molcas_tinker/qmmm_data/melacu63.prm' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_prm'] = filepath

    if 'MOLCAS' in os.environ:
        record['MOLCAS'] = os.environ['MOLCAS']

    if 'TINKER' in os.environ:
        record['TINKER'] = os.environ['TINKER']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |          MOLCAS/TINKER Test Calculation           |
 |                                                   |
 *---------------------------------------------------*

 Some enviroment variables are needed for this test:

    export MOLCAS=/path
    export TINKER=/path

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
 Copy energy test files:
 %-10s --> %s/sp/PSB5.inp (renamed to PSB5.molcas)
 %-10s --> %s/sp/PSB5.key
 %-10s --> %s/sp/PSB5.xyz
 %-10s --> %s/sp/PSB5.txyz
 %-10s --> %s/sp/PSB5.StrOrb

 Copy coupling test files:
 %-10s --> %s/PSB5.inp (renamed to PSB5.molcas)
 %-10s --> %s/PSB5.key
 %-10s --> %s/PSB5.xyz
 %-10s --> %s/PSB5.velocity.xyz
 %-10s --> %s/PSB5.StrOrb

 Run MOLCAS CASSCF:
""" % ('energy', testdir,
       'qmmm_key', testdir,
       'coord', testdir,
       'qmmm_xyz', testdir,
       'orbital', testdir,
       'coupling', testdir,
       'qmmm_key', testdir,
       'qmmm_xyz', testdir,
       'velo', testdir,
       'orbital', testdir)

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

    if os.path.exists('%s/sp' % (testdir)) == False:
        os.makedirs('%s/sp' % (testdir))

    with open(record['qmmm_key'], 'r') as keyfile:
        key = keyfile.read().splitlines()

    key = 'parameters %s\n%s\n' % (record['qmmm_prm'], '\n'.join(key[1:]))

    shutil.copy2(record['energy'], '%s/sp/PSB5.molcas' % (testdir))
    shutil.copy2(record['coord'], '%s/sp/PSB5.xyz' % (testdir))
    shutil.copy2(record['qmmm_xyz'], '%s/sp/PSB5.txyz' % (testdir))
    shutil.copy2(record['orbital'], '%s/sp/PSB5.StrOrb' % (testdir))

    with open('%s/sp/PSB5.key' % (testdir), 'w') as keyfile:
        keyfile.write(key)

    shutil.copy2(record['coupling'], '%s/PSB5.molcas' % (testdir))
    shutil.copy2(record['qmmm_xyz'], '%s/PSB5.xyz' % (testdir))
    shutil.copy2(record['velo'], '%s/PSB5.velocity.xyz' % (testdir))
    shutil.copy2(record['orbital'], '%s/PSB5.StrOrb' % (testdir))

    with open('%s/PSB5.key' % (testdir), 'w') as keyfile:
        keyfile.write(key)

def Setup(record, testdir):
    input_energy = """&CONTROL
title         PSB5
qc_ncpu       1
jobtype       sp
qm            mlctkr

&Molcas
molcas         %s
tinker         %s
molcas_calcdir %s/sp
molcas_mem   5000

&Molecule
ci       2 1
spin     0 1
coupling 1 2, 1 3, 2 3
qmmm_key  %s/sp/PSB5.key
qmmm_xyz %s/sp/PSB5.txyz
""" % ( record['MOLCAS'],
        record['TINKER'],
        testdir,
        testdir,
        testdir)
 
    input_coupling = """&CONTROL
title         PSB5
qc_ncpu       1
jobtype       sp
qm            mlctkr

&Molcas
molcas         %s
tinker         %s
molcas_calcdir %s
molcas_mem   5000

&Molecule
ci       2 1
spin     0 1
coupling 1 2, 1 3, 2 3
qmmm_key  PSB5.key
""" % ( record['MOLCAS'],
        record['TINKER'],
        testdir)

    with open('%s/sp/test_energy' % (testdir), 'w') as out:
        out.write(input_energy)

    with open('%s/test_coupling' % (testdir), 'w') as out:
        out.write(input_coupling)

def Collect(testdir):
    with open('%s/PSB5.log' % (testdir), 'r') as logfile:
        log = logfile.read().splitlines()
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def RunCASPT2(record, testdir, pyrai2mddir):
    maindir = os.getcwd()
    results = ''

    os.chdir('%s/sp' % (testdir))
    subprocess.run('python3 %s/pyrai2md.py test_energy' % (pyrai2mddir), shell = True)
    os.chdir(maindir)
    tmp = Collect('%s/sp' % (testdir))
    results += tmp

    if len(tmp.splitlines()) < 13:
        code = 'FAILED(energy runtime error)'
        return results, code
    else:
        results += ' Energy calculation done, entering coupling calculation... \n'

    os.chdir(testdir)
    subprocess.run('python3 %s/pyrai2md.py test_coupling' % (pyrai2mddir), shell = True)
    os.chdir(maindir)
    tmp = Collect(testdir)
    results += tmp

    if len(tmp.splitlines()) < 13:
        code = 'FAILED(coupling runtime error)'
        return results, code
    else:
        code = 'PASSED'
        results += ' Coupling calculation done\n'

    return results, code
