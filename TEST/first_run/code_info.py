######################################################
#
# PyRAI2MD Code Structure Reveiw
#
# Author Jingbai Li
# Sep 25 2021
#
######################################################

import os

ROOT = '/home/lijingbai2009/share/NN-ChemI/PyRAIMD2/pyrai2md'

register = {
    'pyrai2md'          : 'pyrai2md.py',
    'variables'         : 'PyRAI2MD/variables.py',
    'method'            : 'PyRAI2MD/methods.py',
    'atom'              : 'PyRAI2MD/Molecule/atom.py',
    'molecule'          : 'PyRAI2MD/Molecule/molecule.py',
    'trajectory'        : 'PyRAI2MD/Molecule/trajectory.py',
    'pbc_helper'        : 'PyRAI2MD/Molecule/pbc_helper.py',
    'qmmm_helper'       : 'PyRAI2MD/Molecule/qmmm_helper.py',
    'qc_molcas'         : 'PyRAI2MD/Quantum_Chemistry/qc_molcas.py',
    'qc_bagel'          : 'PyRAI2MD/Quantum_Chemistry/qc_bagel.py',
    'qc_molcas_tinker'  : 'PyRAI2MD/Quantum_Chemistry/qc_molcas_tinker.py',
    'training_data'     : 'PyRAI2MD/Machine_Learning/training_data.py',
    'model_NN'          : 'PyRAI2MD/Machine_Learning/model_NN.py',
    'hypernn'           : 'PyRAI2MD/Machine_Learning/hypernn.py',
    'permutation'       : 'PyRAI2MD/Machine_Learning/permutation.py',
    'adaptive_sampling' : 'PyRAI2MD/Machine_Learning/adaptive_sampling.py',
    'grid_search'       : 'PyRAI2MD/Machine_Learning/grid_search.py',
    'remote_train'      : 'PyRAI2MD/Machine_Learning/remote_train.py',
    'aimd'              : 'PyRAI2MD/Dynamics/aimd.py',
    'mixaimd'           : 'PyRAI2MD/Dynamics/mixaimd.py',
    'single_point'      : 'PyRAI2MD/Dynamics/single_point.py',
    'hop_probability'   : 'PyRAI2MD/Dynamics/hop_probability.py',
    'reset_velocity'    : 'PyRAI2MD/Dynamics/reset_velocity.py',
    'verlet'            : 'PyRAI2MD/Dynamics/verlet.py',
    'ensemble'          : 'PyRAI2MD/Dynamics/Ensembles/ensemble.py',
    'microcanonical'    : 'PyRAI2MD/Dynamics/Ensembles/microcanonical.py',
    'thermostat'        : 'PyRAI2MD/Dynamics/Ensembles/thermostat.py',
    'surface_hopping'   : 'PyRAI2MD/Dynamics/Propagators/surface_hopping.py',
    'fssh'              : 'PyRAI2MD/Dynamics/Propagators/fssh.pyx',
    'gsh'               : 'PyRAI2MD/Dynamics/Propagators/gsh.py',
    'tsh_helper'        : 'PyRAI2MD/Dynamics/Propagators/tsh_helper.py',
    'aligngeom'         : 'PyRAI2MD/Utils/aligngeom.py',
    'coordinates'       : 'PyRAI2MD/Utils/coordinates.py',
    'read_tools'        : 'PyRAI2MD/Utils/read_tools.py',
    'bonds'             : 'PyRAI2MD/Utils/bonds.py',
    'sampling'          : 'PyRAI2MD/Utils/sampling.py',
    'timing'            : 'PyRAI2MD/Utils/timing.py',
    'logo'              : 'PyRAI2MD/Utils/logo.py',
    }

totline = 0
totfile = 0
length = {}
for name, location in register.items():
    mod = '%s/%s' % (ROOT, location)
    totfile += 1
    if os.path.exists(mod) == True:
        with open(mod, 'r') as file:
            n = len(file.readlines())
        totline += n
        length[name] = n
    else:
        length[name] = 0

def review(length, totline, totfile):
    status = """
 File/Folder Name                                  Contents                                      Length
--------------------------------------------------------------------------------------------------------
 pyrai2md.py                                       PyRAI2MD interface                          %8s
 PyRAI2MD                                          source codes folder
  |--variables.py                                  PyRAI2MD input reader                       %8s
  |--method.py                                     PyRAI2MD method manager                     %8s
  |--Molecule                                      atom, molecule, trajectory code folder
  |   |--atom.py                                   atomic properties class                     %8s
  |   |--molecule.py                               molecular properties class                  %8s
  |   |--trajectory.py                             trajectory properties class                 %8s
  |   |--pbc_helper.py                             periodic boundary condition functions       %8s
  |    `-qmmm_helper.py                            qmmm functions                              %8s
  |
  |--Quantum_Chemistry                             quantum chemicial program interface folder
  |   |--qc_molcas.py                              OpenMolcas interface                        %8s
  |   |--qc_bagel.py                               BAGEL interface                             %8s
  |    `-qc_molcas_tinker                          OpenMolcas/Tinker interface                 %8s
  |
  |--Machine_Learning                              machine learning library interface folder
  |   |--training_data.py                          training data manager                       %8s
  |   |--model_NN.py                               neural network interface                    %8s
  |   |--hypernn.py                                hyperparameter manager                      %8s
  |   |--permutation.py                            data permutation functions                  %8s
  |   |--adaptive_sampling.py                      adaptive sampling class                     %8s
  |   |--grid_search.py                            grid search class                           %8s
  |   |--remote_train.py                           distribute remote training                  %8s
  |    `-pyNNsMD                                   neural network library                         (6375)
  |
  |--Dynamics                                      ab initio molecular dynamics code folder
  |   |--aimd.py                                   molecular dynamics class                    %8s
  |   |--mixaimd.py                                ML-QC hybrid molecular dynamics class       %8s
  |   |--single_point.py                           single point calculation                    %8s
  |   |--hop_probability.py                        surface hopping probability calculation     %8s
  |   |--reset_velocity.py                         velocity adjustment functions               %8s
  |   |--verlet.py                                 velocity verlet method                      %8s
  |   |--Ensembles                                 thermodynamics control code folder
  |   |   |--ensemble.py                           thermodynamics ensemble manager             %8s
  |   |   |--microcanonical.py                     microcanonical ensemble                     %8s
  |   |    `-thermostat.py                         canonical ensemble                          %8s
  |   |
  |    `-Propagators                               electronic propagation code folder
  |       |--surface_hopping.py                    surface hopping manager                     %8s
  |       |--fssh.pyx                              fewest switches surface hopping method      %8s
  |       |--gsh.py                                generalized surface hopping method          %8s
  |        `-tsh_helper.py                         trajectory surface hopping tools            %8s
  |
   `-Utils                                         utility folder
      |--aligngeom.py                              geometry aligment and comparison functions  %8s
      |--coordinates.py                            coordinates writing functions               %8s
      |--read_tools.py                             index reader                                %8s
      |--bonds.py                                  bond length library                         %8s
      |--sampling.py                               initial condition sampling functions        %8s
      |--timing.py                                 timing functions                            %8s
       `-logo.py                                   logo and credits                            %8s
--------------------------------------------------------------------------------------------------------
Total %4s/%4s files                                                                          %8s
""" % ( length['pyrai2md'],
        length['variables'],
        length['method'],
        length['atom'],
        length['molecule'],
        length['trajectory'],
        length['pbc_helper'],
        length['qmmm_helper'],
        length['qc_molcas'],
        length['qc_bagel'],
        length['qc_molcas_tinker'],
        length['training_data'],
        length['model_NN'],
        length['hypernn'],
        length['permutation'],
        length['adaptive_sampling'],
        length['grid_search'],
        length['remote_train'],
        length['aimd'],
        length['mixaimd'],
        length['single_point'],
        length['hop_probability'],
        length['reset_velocity'],
        length['verlet'],
        length['ensemble'],
        length['microcanonical'],
        length['thermostat'],
        length['surface_hopping'],
        length['fssh'],
        length['gsh'],
        length['tsh_helper'],
        length['aligngeom'],
        length['coordinates'],
        length['read_tools'],
        length['bonds'],
        length['sampling'],
        length['timing'],
        length['logo'],
        totfile,
        len(length),
        totline)

    return status

def main():
    print(review(length, totline, totfile))

if __name__ == '__main__':
    main()
