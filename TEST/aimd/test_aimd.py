######################################################
#
# PyRAI2MD test AIMD
#
# Author Jingbai Li
# Sep 30 2021
#
######################################################

import os, sys, shutil, json, subprocess

def TestAIMD():
    """ molcas/tinker test

    1. AIMD Molcas fssh
    2. AIMD Molcas gsh soc
    3. AIMD Molcas/tinker gsh

    """

    pyrai2mddir = os.environ['PYRAI2MD']
    testdir = '%s/aimd' % (os.getcwd())
    record = {
        'qm_nac'   : 'FileNotFound',
        'qm_soc'   : 'FileNotFound',
        'qm_orb'   : 'FileNotFound',
        'qm_xyz'   : 'FileNotFound',
        'qm_velo'  : 'FileNotFound',
        'qmmm_eg'  : 'FileNotFound',
        'qmmm_xyz' : 'FileNotFound',
        'qmmm_orb' : 'FileNotFound',
        'qmmm_velo': 'FileNotFound',
        'qmmm_key' : 'FileNotFound',
        'qmmm_prm' : 'FileNotFound',
        'MOLCAS'   : 'VariableNotFound',
        'TINKER'   : 'VariableNotFound',
        }


    filepath = '%s/TEST/aimd/aimd_data/c2h4.inp' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qm_nac'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/c2h4.isc.inp' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qm_soc'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/c2h4.StrOrb' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qm_orb'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/c2h4.xyz' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qm_xyz'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/c2h4.velo' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qm_velo'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/PSB5.inp' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_eg'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/PSB5.xyz' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_xyz'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/PSB5.StrOrb' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_orb'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/PSB5.velo' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_velo'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/PSB5.key' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_key'] = filepath

    filepath = '%s/TEST/aimd/aimd_data/melacu63.prm' % (pyrai2mddir)
    if os.path.exists(filepath):
        record['qmmm_prm'] = filepath

    if 'MOLCAS' in os.environ:
        record['MOLCAS'] = os.environ['MOLCAS']

    if 'TINKER' in os.environ:
        record['TINKER'] = os.environ['TINKER']

    summary = """
 *---------------------------------------------------*
 |                                                   |
 |             AIMD  Test Calculation                |
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
 Copy fssh test files:
 %-10s --> %s/fssh/c2h4.inp (renamed to PSB5.molcas)
 %-10s --> %s/fssh/c2h4.StrOrb
 %-10s --> %s/fssh/c2h4.xyz
 %-10s --> %s/fssh/c2h4.velo

 Copy gsh test file
 %-10s --> %s/gsh_soc/c2h4.inp (renamed to PSB5.molcas)
 %-10s --> %s/gsh_soc/c2h4.StrOrb
 %-10s --> %s/gsh_soc/c2h4.xyz
 %-10s --> %s/gsh_soc/c2h4.velo

 Copy gsh test files:
 %-10s --> %s/gsh/PSB5.inp (renamed to PSB5.molcas)
 %-10s --> %s/gsh/PSB5.key
 %-10s --> %s/gsh/PSB5.xyz
 %-10s --> %s/gsh/PSB5.velo
 %-10s --> %s/gsh/PSB5.StrOrb


 Run MOLCAS CASSCF:
""" % ( 'qm_nac', testdir,
        'qm_orb', testdir,
        'qm_xyz', testdir,
        'qm_velo', testdir,
        'qm_soc', testdir,
        'qm_orb', testdir,
        'qm_xyz', testdir,
        'qm_velo', testdir,
        'qmmm_inp', testdir,
        'qmmm_key', testdir,
        'qmmm_xyz', testdir,
        'qmmm_velo', testdir,
        'qmmm_orb', testdir,
        )

    results, code = RunMD(record, testdir, pyrai2mddir)
   
    summary += """
-------------------------------------------------------
                     AIMD OUTPUT
-------------------------------------------------------
%s
-------------------------------------------------------
""" % (results)   
    return summary, code
 
def CopyInput(record, testdir):

    if os.path.exists('%s/fssh' % (testdir)) == False:
        os.makedirs('%s/fssh' % (testdir))

    if os.path.exists('%s/gsh_soc' % (testdir)) == False:
        os.makedirs('%s/gsh_soc' % (testdir))

    if os.path.exists('%s/gsh' % (testdir)) == False:
        os.makedirs('%s/gsh' % (testdir))

    shutil.copy2(record['qm_nac'], '%s/fssh/c2h4.molcas' % (testdir))
    shutil.copy2(record['qm_xyz'], '%s/fssh/c2h4.xyz' % (testdir))
    shutil.copy2(record['qm_velo'], '%s/fssh/c2h4.velo' % (testdir))
    shutil.copy2(record['qm_orb'], '%s/fssh/c2h4.StrOrb' % (testdir))

    shutil.copy2(record['qm_soc'], '%s/gsh_soc/c2h4.molcas' % (testdir))
    shutil.copy2(record['qm_xyz'], '%s/gsh_soc/c2h4.xyz' % (testdir))
    shutil.copy2(record['qm_velo'], '%s/gsh_soc/c2h4.velo' % (testdir))
    shutil.copy2(record['qm_orb'], '%s/gsh_soc/c2h4.StrOrb' % (testdir))

    with open(record['qmmm_key'], 'r') as keyfile:
        key = keyfile.read().splitlines()

    shutil.copy2(record['qmmm_eg'], '%s/gsh/PSB5.molcas' % (testdir))
    shutil.copy2(record['qmmm_xyz'], '%s/gsh/PSB5.xyz' % (testdir))
    shutil.copy2(record['qmmm_velo'], '%s/gsh/PSB5.velo' % (testdir))
    shutil.copy2(record['qmmm_orb'], '%s/gsh/PSB5.StrOrb' % (testdir))
    key = 'parameters %s\n%s\n' % (record['qmmm_prm'], '\n'.join(key[1:]))
    with open('%s/gsh/PSB5.key' % (testdir), 'w') as keyfile:
        keyfile.write(key)

def Setup(record, testdir):
    input_fssh = """&CONTROL
title         c2h4
jobtype       md
qm            molcas

&Molcas
molcas         %s
molcas_calcdir %s/fssh
molcas_mem   5000

&Molecule
ci       2
spin     0
coupling 1 2

&MD
step 4
size 20.67
temp 300
thermo nvt
sfhp fssh
root 2
""" % ( record['MOLCAS'],
        testdir)
 
    input_gsh_soc = """&CONTROL
title         c2h4
jobtype       md
qm            molcas

&Molcas
molcas         %s
molcas_calcdir %s/gsh_soc
molcas_mem   5000

&Molecule
ci       2 1
spin     0 1
coupling 1 3, 2 3

&MD
step 4
size 20.67
sfhp gsh
root 2
dosoc 1
thermo nve
""" % ( record['MOLCAS'],
        testdir)

    input_gsh = """&CONTROL
title         PSB5
jobtype       md
qm            mlctkr

&Molcas
molcas         %s
tinker         %s
molcas_calcdir %s/gsh
molcas_mem   5000

&Molecule
ci       2
spin     0
qmmm_key  PSB5.key

&MD
step 4
size 20.67
sfhp gsh
root 2
""" % ( record['MOLCAS'],
        record['TINKER'],
        testdir)

    with open('%s/fssh/fssh' % (testdir), 'w') as out:
        out.write(input_fssh)

    with open('%s/gsh_soc/gsh_soc' % (testdir), 'w') as out:
        out.write(input_gsh_soc)

    with open('%s/gsh/gsh' % (testdir), 'w') as out:
        out.write(input_gsh)

def Collect(testdir, title):
    with open('%s/%s.log' % (testdir, title), 'r') as logfile:
        log = logfile.read().splitlines()
    for n, line in enumerate(log):
        if """State order:""" in line:
            results = log[n - 1:]
            break
    results = '\n'.join(results) + '\n'

    return results

def RunMD(record, testdir, pyrai2mddir):
    maindir = os.getcwd()
    results = ''

    os.chdir('%s/fssh' % (testdir))
    subprocess.run('python3 %s/pyrai2md.py fssh' % (pyrai2mddir), shell = True)
    os.chdir(maindir)
    tmp = Collect('%s/fssh' % (testdir), 'c2h4')
    results += tmp

    if len(tmp.splitlines()) < 13:
        code = 'FAILED(fssh md runtime error)'
        return results, code
    else:
        results += ' FSSH MD calculation done, entering GSH MD with SOC calculation... \n'

    os.chdir('%s/gsh_soc' % (testdir))
    subprocess.run('python3 %s/pyrai2md.py gsh_soc' % (pyrai2mddir), shell = True)
    os.chdir(maindir)
    tmp = Collect('%s/gsh_soc' % (testdir), 'c2h4')
    results += tmp

    if len(tmp.splitlines()) < 13:
        code = 'FAILED(gsh soc md runtime error)'
        return results, code
    else:
        results += ' GSH MD with SOC calculation done, entering QMMM GSH MD calculation... \n'

    os.chdir('%s/gsh' % (testdir))
    subprocess.run('python3 %s/pyrai2md.py gsh' % (pyrai2mddir), shell = True)
    os.chdir(maindir)
    tmp = Collect('%s/gsh' % (testdir), 'PSB5')
    results += tmp

    if len(tmp.splitlines()) < 13:
        code = 'FAILED(qmmm gsh md runtime error)'
        return results, code
    else:
        code = 'PASSED'
        results += ' QMMM GSH MD calculation done\n'

    return results, code
