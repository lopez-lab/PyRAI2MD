
######################################################
#
# PyRAI2MD 2 module for reading input keywords
#
# Author Jingbai Li
# Apr 29 2021
#
######################################################

import os, sys
from PyRAI2MD.Utils.read_tools import ReadVal, ReadIndex

def ReadControl(keywords, values):
    ## This function read variables from &control
    keyfunc = {
        'title'                 : ReadVal('s'),
        'ml_ncpu'               : ReadVal('i'),
        'qc_ncpu'               : ReadVal('i'),
        'gl_seed'               : ReadVal('i'),
        'jobtype'               : ReadVal('s'),
        'qm'                    : ReadVal('s'),
        'abinit'                : ReadVal('s'),
        'refine'                : ReadVal('i'),
        'refine_num'            : ReadVal('i'),
        'refine_start'          : ReadVal('i'),
        'refine_end'            : ReadVal('i'),
        'maxiter'               : ReadVal('i'),
        'maxsample'             : ReadVal('i'),
        'dynsample'             : ReadVal('i'),
        'maxdiscard'            : ReadVal('i'),
        'maxenergy'             : ReadVal('f'),
        'minenergy'             : ReadVal('f'),
        'dynenergy'             : ReadVal('f'),
        'inienergy'             : ReadVal('f'),
        'fwdenergy'             : ReadVal('i'),
        'bckenergy'             : ReadVal('i'),
        'maxgrad'               : ReadVal('f'),
        'mingrad'               : ReadVal('f'),
        'dyngrad'               : ReadVal('f'),
        'inigrad'               : ReadVal('f'),
        'fwdgrad'               : ReadVal('i'),
        'bckgrad'               : ReadVal('i'),
        'maxnac'                : ReadVal('f'),
        'minnac'                : ReadVal('f'),
        'dynnac'                : ReadVal('f'),
        'ininac'                : ReadVal('f'),
        'fwdnac'                : ReadVal('i'),
        'bcknac'                : ReadVal('i'),
        'maxsoc'                : ReadVal('f'),
        'minsoc'                : ReadVal('f'),
        'dynsoc'                : ReadVal('f'),
        'inisoc'                : ReadVal('f'),
        'fwdsoc'                : ReadVal('i'),
        'bcksoc'                : ReadVal('i'),
        'load'                  : ReadVal('i'),
        'transfer'              : ReadVal('i'),
        'pop_step'              : ReadVal('i'),
        'verbose'               : ReadVal('i'),
        'silent'                : ReadVal('i'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in $control' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadMolecule(keywords, values):
    ## This function read variables from &molecule
    keyfunc = {
        'ci'                    : ReadVal('il'),
        'spin'                  : ReadVal('il'),
        'coupling'              : ReadIndex('g'),
        'qmmm_key'              : ReadVal('s'),
        'qmmm_xyz'              : ReadVal('s'),
        'highlevel'             : ReadIndex('s'),
        'boundary'              : ReadIndex('g'),
        'freeze'                : ReadIndex('s'),
        'constrain'             : ReadIndex('g'),
        'primitive'             : ReadIndex('g'),
        'lattice'               : ReadIndex('s'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &molecule' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadMolcas(keywords, values):
    ## This function read variables from &molcas
    keyfunc = {
        'molcas'                : ReadVal('s'),
        'molcas_nproc'          : ReadVal('s'),
        'molcas_mem'            : ReadVal('s'),
        'molcas_print'          : ReadVal('s'),
        'molcas_project'        : ReadVal('s'),
        'molcas_calcdir'        : ReadVal('s'),
        'molcas_workdir'        : ReadVal('s'),
        'track_phase'           : ReadVal('i'),
        'basis'                 : ReadVal('i'),
        'omp_num_threads'       : ReadVal('s'),
        'use_hpc'               : ReadVal('i'),
        'keep_tmp'              : ReadVal('i'),
        'verbose'               : ReadVal('i'),
        'tinker'                : ReadVal('s'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &molcas' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadBagel(keywords, values):
    ## This function read variables from &bagel
    keyfunc = {
        'bagel'                 : ReadVal('s'),
        'bagel_nproc'           : ReadVal('s'),
        'bagel_project'         : ReadVal('s'),
        'bagel_workdir'         : ReadVal('s'),
        'bagel_archive'         : ReadVal('s'),
        'mpi'                   : ReadVal('s'),
        'blas'                  : ReadVal('s'),
        'lapack'                : ReadVal('s'),
        'boost'                 : ReadVal('s'),
        'mkl'                   : ReadVal('s'),
        'arch'                  : ReadVal('s'),
        'omp_num_threads'       : ReadVal('s'),
        'use_mpi'               : ReadVal('i'),
        'use_hpc'               : ReadVal('i'),
        'keep_tmp'              : ReadVal('i'),
        'verbose'               : ReadVal('i'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &bagel' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadMD(keywords, values):
    ## This function read variables from &md
    keyfunc = {
        'initdcond'             : ReadVal('i'),
        'excess'                : ReadVal('f'),
        'scale'                 : ReadVal('f'),
        'target'                : ReadVal('f'),
        'graddesc'              : ReadVal('i'),
        'reset'                 : ReadVal('i'),
        'resetstep'             : ReadVal('i'),
        'ninitcond'             : ReadVal('i'),
        'method'                : ReadVal('s'),
        'format'                : ReadVal('s'),
        'temp'                  : ReadVal('i'),
        'step'                  : ReadVal('i'),
        'size'                  : ReadVal('f'),
        'root'                  : ReadVal('i'),
        'activestate'           : ReadVal('i'),
        'sfhp'                  : ReadVal('s'),
        'nactype'               : ReadVal('s'),
        'phasecheck'            : ReadVal('i'),
        'gap'                   : ReadVal('f'),
        'gapsoc'                : ReadVal('f'),
        'substep'               : ReadVal('i'),
        'integrate'             : ReadVal('i'),
        'deco'                  : ReadVal('s'),
        'adjust'                : ReadVal('i'),
        'reflect'               : ReadVal('i'),
        'maxh'                  : ReadVal('i'),
        'dosoc'                 : ReadVal('i'),
        'thermo'                : ReadVal('s'),
        'thermodelay'           : ReadVal('i'),
        'silent'                : ReadVal('i'),
        'verbose'               : ReadVal('i'),
        'direct'                : ReadVal('i'),
        'buffer'                : ReadVal('i'),
        'record'                : ReadVal('i'),
        'checkpoint'            : ReadVal('i'),
        'restart'               : ReadVal('i'),
        'addstep'               : ReadVal('i'),
        'ref_energy'            : ReadVal('i'),
        'ref_grad'              : ReadVal('i'),
        'ref_nac'               : ReadVal('i'),
        'ref_soc'               : ReadVal('i'),
        'datapath'              : ReadVal('s'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &md' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords         

def ReadNN(keywords, values):
    ## This function read variables from &nn
    keyfunc = {
        'train_mode'            : ReadVal('s'),
        'train_data'            : ReadVal('s'),
        'pred_data'             : ReadVal('s'),
        'modeldir'              : ReadVal('s'),
        'nn_eg_type'            : ReadVal('i'),
        'nn_nac_type'           : ReadVal('i'),
        'nn_soc_type'           : ReadVal('i'),
        'shuffle'               : ReadVal('b'),
        'eg_unit'               : ReadVal('s'),
        'nac_unit'              : ReadVal('s'),
        'soc_unit'              : ReadVal('s'),
        'permute_map'           : ReadVal('s'),
        'gpu'                   : ReadVal('i'),
        'silent'                : ReadVal('i'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &nn' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadGrids(keywords, values):
    ## This function read variables form &search
    keyfunc = {
        'depth'                 : ReadVal('il'),
        'nn_size'               : ReadVal('il'),
        'batch_size'            : ReadVal('il'),
        'reg_l1'                : ReadVal('fl'),
        'reg_l2'                : ReadVal('fl'),
        'dropout'               : ReadVal('fl'),
        'use_hpc'               : ReadVal('i'),
        'retrieve'              : ReadVal('i'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &search' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadHyper(keywords, values):
    ## This function read variables from &e1,&e2,&g1,&g2,&eg1,&eg2,&nac1,&nac2 
    keyfunc = {
        'invd_index'            : ReadIndex('g'),
        'angle_index'           : ReadIndex('g'),
        'dihed_index'           : ReadIndex('g'),
        'depth'                 : ReadVal('i'),
        'nn_size'               : ReadVal('i'),
        'activ'                 : ReadVal('s'),
        'activ_alpha'           : ReadVal('f'),
        'loss_weights'          : ReadVal('fl'),
        'use_dropout'           : ReadVal('b'),
        'dropout'               : ReadVal('f'),
        'use_reg_activ'         : ReadVal('s'),
        'use_reg_weight'        : ReadVal('s'),
        'use_reg_bias'          : ReadVal('s'),
        'reg_l1'                : ReadVal('f'),
        'reg_l2'                : ReadVal('f'),
        'use_step_callback'     : ReadVal('b'),
        'use_linear_callback'   : ReadVal('b'),
        'use_early_callback'    : ReadVal('b'),
        'use_exp_callback'      : ReadVal('b'),
        'scale_x_mean'          : ReadVal('b'),
        'scale_x_std'           : ReadVal('b'),
        'scale_y_mean'          : ReadVal('b'),
        'scale_y_std'           : ReadVal('b'),
        'normalization_mode'    : ReadVal('i'),
        'learning_rate'         : ReadVal('f'),
        'phase_less_loss'       : ReadVal('b'),
        'initialize_weights'    : ReadVal('b'),
        'val_disjoint'          : ReadVal('b'),
        'val_split'             : ReadVal('f'),
        'epo'                   : ReadVal('i'),
        'epomin'                : ReadVal('i'),
        'pre_epo'               : ReadVal('i'),
        'patience'              : ReadVal('i'),
        'max_time'              : ReadVal('i'),
        'batch_size'            : ReadVal('i'),
        'delta_loss'            : ReadVal('f'),
        'loss_monitor'          : ReadVal('s'),
        'factor_lr'             : ReadVal('f'),
        'epostep'               : ReadVal('i'),
        'learning_rate_start'   : ReadVal('f'),
        'learning_rate_stop'    : ReadVal('f'),
        'learning_rate_step'    : ReadVal('fl'),
        'epoch_step_reduction'  : ReadVal('il'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &eg/&nac/&soc' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadFile(keywords, values):
    ## This function read variables form &file
    keyfunc = {
        'natom'                 : ReadVal('i'),
        'file'                  : ReadVal('s'),
        }

    for i in values:
        if len(i.split()) < 2:
            continue
        key, val = i.split()[0], i.split()[1:]
        key = key.lower()
        if key not in keyfunc.keys():
            sys.exit('\n  KeywordError\n  PyRAI2MD: cannot recognize keyword %s in &file' % (key))
        keywords[key] = keyfunc[key](val)

    return keywords

def ReadInput(input):
    ## This function store all default values for variables
    ## This fucntion read variable from input
    ## This function is expected to be expanded in future as more methods added

    ## default values
    variables_control = {
        'title'                 : None,
        'ml_ncpu'               : 1,
        'qc_ncpu'               : 1,
        'gl_seed'               : 1,
        'jobtype'               :'sp',
        'qm'                    :'nn',
        'abinit'                :'molcas',
        'refine'                : 0,
        'refine_num'            : 4,
        'refine_start'          : 0,
        'refine_end'            : 200,
        'maxiter'               : 1,
        'maxsample'             : 1,
        'dynsample'             : 0,
        'maxdiscard'            : 0,
        'maxenergy'             : 0.05,
        'minenergy'             : 0.02,
        'dynenergy'             : 0.1,
        'inienergy'             : 0.3,
        'fwdenergy'             : 1,
        'bckenergy'    	       	: 1,
        'maxgrad'               : 0.15,
        'mingrad'               : 0.06,
        'dyngrad'               : 0.1,
        'inigrad'               : 0.3,
        'fwdgrad'               : 1,
        'bckgrad'               : 1,
        'maxnac'                : 0.15,
        'minnac'                : 0.06,
        'dynnac'                : 0.1,
        'ininac'                : 0.3,
        'fwdnac'                : 1,
        'bcknac'                : 1,
        'maxsoc'                : 50,
        'minsoc'                : 20,
        'dynsoc'                : 0.1,
        'inisoc'                : 0.3,
        'fwdsoc'                : 1,
        'bcksoc'                : 1,
        'load'                  : 1,
        'transfer'              : 0,
        'pop_step'              : 200,
        'verbose'               : 2,
        'silent'                : 1,
        }

    variables_molecule = {
        'qmmm_key'              : None,
        'qmmm_xyz'              :'Input',
        'ci'                    :[1],
        'spin'                  :[0],
        'coupling'              :[],
        'highlevel'             :[],
        'boundary'              :[],
        'freeze'                :[],
        'constrain'             :[],
        'primitive'             :[],
        'lattice'               :[],
    }

    variables_molcas = {
        'molcas'                :'/work/lopez/Molcas',
        'molcas_nproc'          :'1',
        'molcas_mem'            :'2000',
        'molcas_print'          :'2',
        'molcas_project'        : None,
        'molcas_calcdir'        : os.getcwd(),
        'molcas_workdir'        : None,
        'track_phase'           : 0,
        'basis'                 : 2,
        'omp_num_threads'       :'1',
        'use_hpc'               : 0,
        'group'                 : None,  # Caution! Not allow user to set.
        'keep_tmp'              : 1,
        'verbose'               : 0,
        'tinker'                : '/work/lopez/Molcas/tinker-6.3.3/bin',
        }

    variables_bagel = {
        'bagel'                 :'/work/lopez/Bagel-mvapich',
        'bagel_nproc'           : 1,
        'bagel_project'         : None,
        'bagel_workdir'         : os.getcwd(),
        'bagel_archive'         :'default',
        'mpi'                   :'/work/lopez/mvapich2-2.3.4',
        'blas'                  :'/work/lopez/BLAS',
        'lapack'                :'/work/lopez/BLAS',
        'boost'                 :'/work/lopez/Boost',
        'mkl'                   :'/work/lopez/intel/mkl/bin/mklvars.sh',
        'arch'                  :'intel64',
        'omp_num_threads'       :'1',
        'use_mpi'               : 0,
        'use_hpc'  	        : 0,
        'group'                 : None,  # Caution! Not allow user to set.
        'keep_tmp'              : 1,
        'verbose'               : 0,
        }

    variables_md = {
        'gl_seed'               : 1,     # Caution! Not allow user to set.
        'initcond'              : 0,
        'excess'                : 0,
        'scale'                 : 1,
        'target'                : 0,
        'graddesc'              : 0,
        'reset'                 : 0,
        'resetstep'             : 0,
        'ninitcond'             : 20,
        'method'                :'wigner',
        'format'                :'molden',
        'temp'                  : 300,
        'step'                  : 10,
        'size'                  : 20.67,
        'root'                  : 1,
        'activestate'           : 0,
        'sfhp'                  :'nosh',
        'nactype'               :'ktdc',
        'phasecheck'            : 0,
        'gap'                   : 0.5,
        'gapsoc'                : 0.5,
        'substep'               : 20,
        'integrate'             : 0,
        'deco'                  :'0.1',
        'adjust'                : 1,
        'reflect'               : 1,
        'maxh'                  : 10,
        'dosoc'                 : 0,
        'thermo'                :'off',
        'thermodelay'           : 200,
        'silent'                : 1,
        'verbose'               : 0,
        'direct'                : 2000,
        'buffer'                : 500,
        'record'                : 0,
        'checkpoint'            : 0,
        'restart'               : 0,
        'addstep'               : 0,
        'group'                 : None,    # Caution! Not allow user to set.
        'ref_energy'            : 0,
        'ref_grad'              : 0,
        'ref_nac'               : 0, 
        'ref_soc'               : 0,
        'datapath'              : None,
        }

    variables_nn = {
        'train_mode'            :'training',
        'train_data'            : None,
        'pred_data'             : None,
        'modeldir'              : None,
        'silent'                : 1,
        'nn_eg_type'            : 1,
        'nn_nac_type'           : 0,
        'nn_soc_type'           : 0,
        'shuffle'               : False,
        'eg_unit'               :'si',
        'nac_unit'              :'si',
        'soc_unit'              :'si',
        'ml_seed'               : 1,     # Caution! Not allow user to set.
        'data'                  : None,  # Caution! Not allow user to set.
        'search'                : None,  # Caution! Not allow user to set.
        'eg'                    : None,  # Caution! This value will be updated later. Not allow user to set.
        'nac'                   : None,  # Caution! This value will be updated later. Not allow user to set.
        'eg2'                   : None,  # Caution! This value will be updated later. Not allow user to set.
        'nac2'                  : None,  # Caution! This value will be updated later. Not allow user to set.
        'soc'                   : None,  # Caution! This value will be updated later. Not allow user to set.
        'soc2'                  : None,  # Caution! This value will be updated later. Not allow user to set.
        'permute_map'           :'No',
        'gpu'                   : 0,
        }

    variables_search = {
        'depth'                 :[],
        'nn_size'               :[],
        'batch_size'            :[],
        'reg_l1'                :[],
        'reg_l2'                :[],
        'dropout'               :[],
        'use_hpc'               : 0,
        'retrieve'              : 0,
        }

    variables_eg = {
        'model_type'            :'mlp_eg',
        'invd_index'            :[],
        'angle_index'           :[],
        'dihed_index'           :[],
        'depth'                 : 4,
        'nn_size'               : 100,
        'activ'                 :'leaky_softplus',
        'activ_alpha'           : 0.03,
        'loss_weights'          :[1, 1],
        'use_dropout'           : False,
        'dropout'               : 0.005,
        'use_reg_activ'    	: None,
        'use_reg_weight'        : None,
        'use_reg_bias'          : None,
        'reg_l1'                : 1e-5,
        'reg_l2'                : 1e-5,
        'use_step_callback'     : True,
        'use_linear_callback'   : False,
        'use_early_callback'    : False,
        'use_exp_callback'      : False,
        'scale_x_mean'          : False,
        'scale_x_std'           : False,
        'scale_y_mean'          : True,
        'scale_y_std'           : True,
        'normalization_mode'    : 1,
        'learning_rate'         : 1e-3,
        'initialize_weights'    : True,
        'val_disjoint'          : True,
        'val_split'             : 0.1,
        'epo'                   : 2000,
        'epomin'                : 1000,
        'patience'              : 300,
        'max_time'              : 300,
        'batch_size'            : 64,
        'delta_loss'            : 1e-5,
        'loss_monitor'          :'val_loss',
        'factor_lr'             : 0.1,
        'epostep'               : 10,
        'learning_rate_start'   : 1e-3,
        'learning_rate_stop'    : 1e-6,
        'learning_rate_step'    : [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction'  : [500, 500, 500, 500],
        }

    variables_nac = {
        'model_type'            :'mlp_nac2',
        'invd_index'       	:[],
        'angle_index'           :[],
        'dihed_index'           :[],
        'depth'                 : 4,
        'nn_size'               : 100,
        'activ'                 :'leaky_softplus',
        'activ_alpha'           : 0.03,
        'use_dropout'      	: False,
        'dropout'  	       	: 0.005,
        'use_reg_activ'         : None,
        'use_reg_weight'        : None,
        'use_reg_bias'          : None,
        'reg_l1'                : 1e-5,
        'reg_l2'                : 1e-5,
        'use_step_callback'	: True,
        'use_linear_callback'   : False,
        'use_early_callback'    : False,
        'use_exp_callback' 	: False,
        'scale_x_mean'          : False,
        'scale_x_std'           : False,
        'scale_y_mean'          : True,
        'scale_y_std'           : True,
        'normalization_mode'    : 1,
        'learning_rate'         : 1e-3,
        'phase_less_loss'       : False,
        'initialize_weights'    : True,
        'val_disjoint'          : True,
        'val_split'             : 0.1,
        'epo'                   : 2000,
        'epomin'                : 1000,
        'pre_epo'               : 100,
        'patience'              : 300,
        'max_time'              : 300,
        'batch_size'            : 64,
        'delta_loss'            : 1e-5,
        'loss_monitor'          :'val_loss',
        'factor_lr'             : 0.1,
        'epostep'               : 10,
        'learning_rate_start'   : 1e-3,
        'learning_rate_stop'    : 1e-6,
        'learning_rate_step'    : [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction'  : [500, 500, 500, 500],
        }

    variables_soc = {
        'model_type'            :'mlp_e',
        'invd_index'            :[],
        'angle_index'           :[],
        'dihed_index'           :[],
        'depth'                 : 4,
        'nn_size'               : 100,
        'activ'                 :'leaky_softplus',
        'activ_alpha'           : 0.03,
        'use_dropout'           : False,
        'dropout'               : 0.005,
        'use_reg_activ'         : None,
        'use_reg_weight'        : None,
        'use_reg_bias'          : None,
        'reg_l1'                : 1e-5,
        'reg_l2'                : 1e-5,
        'use_step_callback'     : True,
        'use_linear_callback'   : False,
        'use_early_callback'    : False,
        'use_exp_callback'      : False,
        'scale_x_mean'          : False,
        'scale_x_std'           : False,
        'scale_y_mean'          : True,
        'scale_y_std'           : True,
        'normalization_mode'    : 1,
        'learning_rate'         : 1e-3,
        'initialize_weights'    : True,
        'val_disjoint'          : True,
        'val_split'             : 0.1,
        'epo'                   : 2000,
        'epomin'                : 1000,
        'patience'              : 300,
        'max_time'              : 300,
        'batch_size'            : 64,
        'delta_loss'            : 1e-5,
        'loss_monitor'          :'val_loss',
        'factor_lr'             : 0.1,
        'epostep'               : 10,
        'learning_rate_start'   : 1e-3,
        'learning_rate_stop'    : 1e-6,
        'learning_rate_step'    : [1e-3, 1e-4, 1e-5, 1e-6],
        'epoch_step_reduction'  : [500, 500, 500, 500],
        }

    variables_file = {
        'natom'                 : 0,
        'file'                  : None,
        }

    ## More default will be added below

    ## ready to read input
    variables_input = {
        'control'               : variables_control,
        'molecule'              : variables_molecule,
        'molcas'                : variables_molcas,
        'bagel'                 : variables_bagel,
        'md'                    : variables_md,
        'nn'                    : variables_nn,
        'search'                : variables_search,
        'eg'                    : variables_eg.copy(),
        'nac'                   : variables_nac.copy(),
        'soc'                   : variables_soc.copy(),
        'eg2'                   : variables_eg.copy(),
        'nac2'                  : variables_nac.copy(),
        'soc2'                  : variables_soc.copy(),
        'file'                  : variables_file.copy(),
        }
  
    variables_readfunc = {
        'control'               : ReadControl,
        'molecule'              : ReadMolecule,
        'molcas'                : ReadMolcas,
        'bagel'                 : ReadBagel,
        'md'                    : ReadMD,
        'nn'                    : ReadNN,
        'search'                : ReadGrids,
        'eg'                    : ReadHyper,
        'nac'                   : ReadHyper,
        'soc'                   : ReadHyper,
        'eg2'                   : ReadHyper,
        'nac2'                  : ReadHyper,
        'soc2'                  : ReadHyper,
        'file'                  : ReadFile,
        }

    ## read variable if input is a list of string
    ## skip reading if input is a json dict
    if isinstance(input, list):
        for line in input:
            line = line.splitlines()
            if len(line) == 0:
                continue
            variable_name = line[0].lower()
            variables_input[variable_name] = variables_readfunc[variable_name](variables_input[variable_name], line)

    ## assemble variables
    variables_all={
        'control'               : variables_input['control'],
        'molecule'              : variables_input['molecule'],
        'molcas'                : variables_input['molcas'],
        'bagel'                 : variables_input['bagel'],
        'md'                    : variables_input['md'],
        'nn'                    : variables_input['nn'],
        'file'                  : variables_input['file']
        }

    ## update variables_nn
    variables_all['nn']['search'] = variables_input['search']
    variables_all['nn']['eg']     = variables_input['eg']
    variables_all['nn']['nac']    = variables_input['nac']
    variables_all['nn']['soc']    = variables_input['soc']
    variables_all['nn']['eg2']    = variables_input['eg2']
    variables_all['nn']['nac2']   = variables_input['nac2']
    variables_all['nn']['soc2']   = variables_input['soc2']

    variables_all['md']['gl_seed']            = variables_all['control']['gl_seed']
    variables_all['nn']['ml_seed']            = variables_all['control']['gl_seed']
    variables_all['molcas']['molcas_project'] = variables_all['control']['title']
    variables_all['molcas']['verbose']        = variables_all['md']['verbose']
    variables_all['bagel']['bagel_project']   = variables_all['control']['title']
    variables_all['bagel']['verbose']         = variables_all['md']['verbose']

    ## update variables if input is a dict
    ## be caution that the input dict must have the same data structure
    ## this is only use to load pre-stored input in json
    if isinstance(input, dict):
        variables_all = DeepUpdate(variables_all, input)

    return variables_all

def DeepUpdate(a, b):
    ## recursively update a with b
    for key, val in b.items():
        if key in a.keys():
            if isinstance(val, dict) and isinstance(a[key], dict):
                a[key] = DeepUpdate(a[key], val)
            else:
                a[key] = val
        else:
            a[key] = val

    return a

def StartInfo(variables_all):
    ##  This funtion print start information 

    variables_control  = variables_all['control']
    variables_molecule = variables_all['molecule']
    variables_molcas   = variables_all['molcas']
    variables_bagel    = variables_all['bagel']
    variables_md       = variables_all['md']
    variables_nn       = variables_all['nn']
    variables_eg       = variables_nn['eg']
    variables_nac      = variables_nn['nac']
    variables_soc      = variables_nn['soc']
    variables_eg2      = variables_nn['eg2']
    variables_nac2     = variables_nn['nac2']
    variables_soc2     = variables_nn['soc2']
    variables_search   = variables_nn['search']

    control_info = """
  &control
-------------------------------------------------------
  Title:                      %-10s
  NCPU for ML:                %-10s
  NCPU for QC:                %-10s
  Seed:                       %-10s
  Job: 	                      %-10s
  QM:          	       	      %-10s
  Ab initio:                  %-10s
-------------------------------------------------------

""" % ( variables_control['title'],
        variables_control['ml_ncpu'],
        variables_control['qc_ncpu'],
        variables_control['gl_seed'],
        variables_control['jobtype'],
        variables_control['qm'],
        variables_control['abinit'])

    molecule_info = """
  &molecule
-------------------------------------------------------
  States:                     %-10s
  Spin:                       %-10s
  Interstates:                %-10s
  QMMM keyfile:               %-10s
  QMMM xyzfile:               %-10s
  High level region:          %-10s ...
  Boundary:                   %-10s ...
  Frozen atoms:               %-10s
  Constrained atoms:          %-10s
  Primitive vectors:          %-10s
  Lattice constant:           %-10s
-------------------------------------------------------

""" % ( variables_molecule['ci'],
        variables_molecule['spin'],
        variables_molecule['coupling'],
        variables_molecule['qmmm_key'],
        variables_molecule['qmmm_xyz'],
        variables_molecule['highlevel'][0:10],
        variables_molecule['boundary'][0:5],
        variables_molecule['freeze'],
        variables_molecule['constrain'],
        variables_molecule['primitive'],
        variables_molecule['lattice'])

    adaptive_info = """
  &adaptive sampling method
-------------------------------------------------------
  Ab initio:                  %-10s
  Load trained model:         %-10s
  Transfer learning:          %-10s
  Maxiter:                    %-10s
  Sampling number per traj:   %-10s
  Use dynamical Std:          %-10s
  Max discard range           %-10s
  Refine crossing:            %-10s
  Refine points/range: 	      %-10s %-10s %-10s
  MaxStd  energy:             %-10s
  MinStd  energy:             %-10s
  InitStd energy:             %-10s
  Dynfctr energy:             %-10s
  Forward delay energy:       %-10s
  Backward delay energy:      %-10s
  MaxStd  gradient:           %-10s
  MinStd  gradient:           %-10s
  InitStd gradient:           %-10s
  Dynfctr gradient:           %-10s
  Forward delay	gradient:     %-10s
  Backward delay gradient:    %-10s
  MaxStd  nac:                %-10s
  MinStd  nac:                %-10s
  InitStd nac:                %-10s
  Dynfctr nac:                %-10s
  Forward delay	nac:          %-10s
  Backward delay nac:         %-10s
  MaxStd  soc:                %-10s
  MinStd  soc:                %-10s
  InitStd soc:                %-10s
  Dynfctr soc:                %-10s
  Forward delay	soc:   	      %-10s
  Backward delay soc:  	      %-10s
-------------------------------------------------------

""" % ( variables_control['abinit'],
        variables_control['load'],
        variables_control['transfer'],
        variables_control['maxiter'],
        variables_control['maxsample'],
        variables_control['dynsample'],
        variables_control['maxdiscard'],
        variables_control['refine'],
        variables_control['refine_num'],
        variables_control['refine_start'],
        variables_control['refine_end'],
        variables_control['maxenergy'],
        variables_control['minenergy'],
        variables_control['inienergy'],
        variables_control['dynenergy'],
        variables_control['fwdenergy'],
        variables_control['bckenergy'],
        variables_control['maxgrad'],
        variables_control['mingrad'],
        variables_control['inigrad'],
        variables_control['dyngrad'],
        variables_control['fwdgrad'],
        variables_control['bckgrad'],
        variables_control['maxnac'],
        variables_control['minnac'],
        variables_control['ininac'],
        variables_control['dynnac'],
        variables_control['fwdnac'],
        variables_control['bcknac'],
        variables_control['maxsoc'],
        variables_control['minsoc'],
        variables_control['inisoc'],
        variables_control['dynsoc'],
        variables_control['fwdsoc'],
        variables_control['bcksoc'])

    md_info = """
  &initial condition
-------------------------------------------------------
  Generate initial condition: %-10s
  Number:                     %-10s
  Method:                     %-10s 
  Format:                     %-10s
-------------------------------------------------------

""" % ( variables_md['initcond'],
        variables_md['ninitcond'],
        variables_md['method'],
        variables_md['format'])
    md_info += """
  &md
-------------------------------------------------------
  Initial state:              %-10s
  Temperature (K):            %-10s
  Step:                       %-10s
  Dt (au):                    %-10s
  Only active state grad      %-10s
  Surface hopping:            %-10s
  NAC type:                   %-10s
  Phase correction            %-10s
  Substep:                    %-10s
  Integrate probability       %-10s
  Decoherance:                %-10s
  Adjust velocity:            %-10s
  Reflect velocity:           %-10s
  Maxhop:                     %-10s
  Thermodynamic:              %-10s
  Thermodynamic delay:        %-10s
  Print level:                %-10s
  Direct output:              %-10s
  Buffer output:              %-10s
  Record MD steps:            %-10s
  Checkpoint steps:           %-10s 
  Restart function:           %-10s
  Additional steps:           %-10s
-------------------------------------------------------

""" % ( variables_md['root'],
        variables_md['temp'],
        variables_md['step'],
        variables_md['size'],
        variables_md['activestate'],
        variables_md['sfhp'],
        variables_md['nactype'],
        variables_md['phasecheck'],
        variables_md['substep'],
        variables_md['integrate'],
        variables_md['deco'],
        variables_md['adjust'],
        variables_md['reflect'],
        variables_md['maxh'],
        variables_md['thermo'],
        variables_md['thermodelay'],
        variables_md['verbose'],
        variables_md['direct'],
        variables_md['buffer'],
        variables_md['record'],
        variables_md['checkpoint'],
        variables_md['restart'],
        variables_md['addstep'])

    md_info += """
  &md velocity control
-------------------------------------------------------
  Excess kinetic energy       %-10s
  Scale kinetic energy        %-10s
  Target kinetic energy       %-10s
  Gradient descent path       %-10s
  Reset velocity:             %-10s
  Reset step:                 %-10s
-------------------------------------------------------

""" % ( variables_md['excess'],
        variables_md['scale'],
        variables_md['target'],
        variables_md['graddesc'],
        variables_md['reset'],
        variables_md['resetstep'])

    hybrid_info = """
  &hybrid namd
-------------------------------------------------------
  Mix Energy                  %-10s
  Mix Gradient                %-10s
  Mix NAC                     %-10s
  Mix SOC                     %-10s
-------------------------------------------------------

""" % (variables_md['ref_energy'],
       variables_md['ref_grad'],
       variables_md['ref_nac'],
       variables_md['ref_soc'])

    nn_info = """
  &nn
-------------------------------------------------------
  Train data:                 %-10s
  Predition data:             %-10s
  Train mode:                 %-10s
  Silent mode:                %-10s
  NN EG type:                 %-10s
  NN NAC type:                %-10s
  NN SOC type:                %-10s
  Shuffle data:               %-10s
  EG unit:                    %-10s
  NAC unit:                   %-10s
  Data permutation            %-10s
-------------------------------------------------------

""" % ( variables_nn['train_data'],
        variables_nn['pred_data'],
        variables_nn['train_mode'],
        variables_nn['silent'],
        variables_nn['nn_eg_type'],
        variables_nn['nn_nac_type'],
        variables_nn['nn_soc_type'],
        variables_nn['shuffle'],
        variables_nn['eg_unit'],
        variables_nn['nac_unit'],
        variables_nn['permute_map'])

    nn_info += """
  &hyperparameters            Energy+Gradient      Nonadiabatic         Spin-orbit
----------------------------------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s 
  Angle features:             %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s
  Layers:      	              %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s
  L2:         	              %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Validation disjoint:        %-20s %-20s %-20s
  Validation split:           %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
  Shift_X:     	       	      %-20s %-20s %-20s
  Scale_X:                    %-20s %-20s %-20s
  Shift_Y:                    %-20s %-20s %-20s
  Scale_Y:                    %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % ( len(variables_eg['invd_index']),
        len(variables_nac['invd_index']),
        len(variables_soc['invd_index']),
        len(variables_eg['angle_index']),
        len(variables_nac['angle_index']),
        len(variables_soc['angle_index']),
        len(variables_eg['dihed_index']),
        len(variables_nac['dihed_index']),
        len(variables_soc['dihed_index']),
        variables_eg['activ'],
        variables_nac['activ'],
        variables_soc['activ'],
        variables_eg['activ_alpha'],
        variables_nac['activ_alpha'],
        variables_soc['activ_alpha'],
        variables_eg['depth'],
        variables_nac['depth'],
        variables_soc['depth'],
        variables_eg['nn_size'],
        variables_nac['nn_size'],
        variables_soc['nn_size'],
        variables_eg['use_dropout'],
        variables_nac['use_dropout'],
        variables_soc['use_dropout'],
        variables_eg['dropout'],
        variables_nac['dropout'],
        variables_soc['dropout'],
        variables_eg['use_reg_activ'],
        variables_nac['use_reg_activ'],
        variables_soc['use_reg_activ'],
        variables_eg['use_reg_weight'],
        variables_nac['use_reg_weight'],
        variables_soc['use_reg_weight'],
        variables_eg['use_reg_bias'],
        variables_nac['use_reg_bias'],
        variables_soc['use_reg_bias'],
        variables_eg['reg_l1'],
        variables_nac['reg_l1'],
        variables_soc['reg_l1'],
        variables_eg['reg_l2'],
        variables_nac['reg_l2'],
        variables_soc['reg_l2'],
        variables_eg['loss_weights'],
        '',
        '',
        '',
        variables_nac['phase_less_loss'],
        '',
        variables_eg['initialize_weights'],
        variables_nac['initialize_weights'],
        variables_soc['initialize_weights'],
        variables_eg['val_disjoint'],
        variables_nac['val_disjoint'],
        variables_soc['val_disjoint'],
        variables_eg['val_split'],
        variables_nac['val_split'],
        variables_soc['val_split'],
        variables_eg['epo'],
        variables_nac['epo'],
        variables_soc['epo'],
        '',
        variables_nac['pre_epo'],
        '',
        variables_eg['epomin'],
        variables_nac['epomin'],
        variables_soc['epomin'],
        variables_eg['patience'],
        variables_nac['patience'],
        variables_soc['patience'],
        variables_eg['max_time'],
        variables_nac['max_time'],
        variables_soc['max_time'],
        variables_eg['epostep'],
        variables_nac['epostep'],
        variables_soc['epostep'],
        variables_eg['batch_size'],
        variables_nac['batch_size'],
        variables_soc['batch_size'],
        variables_eg['delta_loss'],
        variables_nac['delta_loss'],
        variables_soc['delta_loss'],
        variables_eg['scale_x_mean'],
        variables_nac['scale_x_mean'],
        variables_soc['scale_x_mean'],
        variables_eg['scale_x_std'],
        variables_nac['scale_x_std'],
        variables_soc['scale_x_std'],
        variables_eg['scale_y_mean'],
        variables_nac['scale_y_mean'],
        variables_soc['scale_y_mean'],
        variables_eg['scale_y_std'],
        variables_nac['scale_y_std'],
        variables_soc['scale_y_std'],
        variables_eg['normalization_mode'],
        variables_nac['normalization_mode'],
        variables_soc['normalization_mode'])

    nn_info += """
  &hyperparameters            Energy+Gradient(2)   Nonadiabatic(2)      Spin-orbit(2)
----------------------------------------------------------------------------------------------
  InvD features:              %-20s %-20s %-20s
  Angle features:             %-20s %-20s %-20s
  Dihedral features:          %-20s %-20s %-20s
  Activation:                 %-20s %-20s %-20s
  Activation alpha:           %-20s %-20s %-20s
  Layers:                     %-20s %-20s %-20s
  Neurons/layer:              %-20s %-20s %-20s
  Dropout:                    %-20s %-20s %-20s
  Dropout ratio:              %-20s %-20s %-20s
  Regularization activation:  %-20s %-20s %-20s
  Regularization weight:      %-20s %-20s %-20s
  Regularization bias:        %-20s %-20s %-20s
  L1:                         %-20s %-20s %-20s
  L2:                         %-20s %-20s %-20s
  Loss weights:               %-20s %-20s %-20s
  Phase-less loss:            %-20s %-20s %-20s
  Initialize weight:          %-20s %-20s %-20s
  Validation disjoint:        %-20s %-20s %-20s
  Validation split:           %-20s %-20s %-20s
  Epoch:                      %-20s %-20s %-20s
  Epoch_pre:                  %-20s %-20s %-20s
  Epoch_min                   %-20s %-20s %-20s
  Patience:                   %-20s %-20s %-20s
  Max time:                   %-20s %-20s %-20s
  Epoch step:                 %-20s %-20s %-20s
  Batch:                      %-20s %-20s %-20s
  Delta loss:                 %-20s %-20s %-20s
  Shift_X:                    %-20s %-20s %-20s
  Scale_X:                    %-20s %-20s %-20s
  Shift_Y:                    %-20s %-20s %-20s
  Scale_Y:                    %-20s %-20s %-20s
  Feature normalization:      %-20s %-20s %-20s
----------------------------------------------------------------------------------------------

""" % ( len(variables_eg2['invd_index']),
        len(variables_nac2['invd_index']),
        len(variables_soc2['invd_index']),
        len(variables_eg2['angle_index']),
        len(variables_nac2['angle_index']),
        len(variables_soc2['angle_index']),
        len(variables_eg2['dihed_index']),
        len(variables_nac2['dihed_index']),
        len(variables_soc2['dihed_index']),
        variables_eg2['activ'],
        variables_nac2['activ'],
        variables_soc2['activ'],
        variables_eg2['activ_alpha'],
        variables_nac2['activ_alpha'],
        variables_soc2['activ_alpha'],
        variables_eg2['depth'],
        variables_nac2['depth'],
        variables_soc2['depth'],
        variables_eg2['nn_size'],
        variables_nac2['nn_size'],
        variables_soc2['nn_size'],
        variables_eg2['use_dropout'],
        variables_nac2['use_dropout'],
        variables_soc2['use_dropout'],
        variables_eg2['dropout'],
        variables_nac2['dropout'],
        variables_soc2['dropout'],
        variables_eg2['use_reg_activ'],
        variables_nac2['use_reg_activ'],
        variables_soc2['use_reg_activ'],
        variables_eg2['use_reg_weight'],
        variables_nac2['use_reg_weight'],
        variables_soc2['use_reg_weight'],
        variables_eg2['use_reg_bias'],
        variables_nac2['use_reg_bias'],
        variables_soc2['use_reg_bias'],
        variables_eg2['reg_l1'],
        variables_nac2['reg_l1'],
        variables_soc2['reg_l1'],
        variables_eg2['reg_l2'],
        variables_nac2['reg_l2'],
        variables_soc2['reg_l2'],
        variables_eg['loss_weights'],
        '',
        '',
        '',
        variables_nac2['phase_less_loss'],
        '',
        variables_eg2['initialize_weights'],
        variables_nac2['initialize_weights'],
        variables_soc2['initialize_weights'],
        variables_eg2['val_disjoint'],
        variables_nac2['val_disjoint'],
        variables_soc2['val_disjoint'],
        variables_eg2['val_split'],
        variables_nac2['val_split'],
        variables_soc2['val_split'],
        variables_eg2['epo'],
        variables_nac2['epo'],
        variables_soc2['epo'],
        '',
        variables_nac2['pre_epo'],
        '',
        variables_eg2['epomin'],
        variables_nac2['epomin'],
        variables_soc2['epomin'],
        variables_eg2['patience'],
        variables_nac2['patience'],
        variables_soc2['patience'],
        variables_eg2['max_time'],
        variables_nac2['max_time'],
        variables_soc2['max_time'],
        variables_eg2['epostep'],
        variables_nac2['epostep'],
        variables_soc2['epostep'],
        variables_eg2['batch_size'],
        variables_nac2['batch_size'],
        variables_soc2['batch_size'],
        variables_eg2['delta_loss'],
        variables_nac2['delta_loss'],
        variables_soc2['delta_loss'],
        variables_eg2['scale_x_mean'],
        variables_nac2['scale_x_mean'],
        variables_soc2['scale_x_mean'],
        variables_eg2['scale_x_std'],
        variables_nac2['scale_x_std'],
        variables_soc2['scale_x_std'],
        variables_eg2['scale_y_mean'],
        variables_nac2['scale_y_mean'],
        variables_soc2['scale_y_mean'],
        variables_eg2['scale_y_std'],
        variables_nac2['scale_y_std'],
        variables_soc2['scale_y_std'],
        variables_eg2['normalization_mode'],
        variables_nac2['normalization_mode'],
        variables_soc2['normalization_mode'])

    search_info = """
  &grid search
-------------------------------------------------------
  Layers:                     %-10s
  Neurons/layer::             %-10s
  Batch:                      %-10s
  L1:                         %-10s
  L2:                         %-10s
  Dropout:                    %-10s
  Job distribution            %-10s
  Retrieve data               %-10s
-------------------------------------------------------

""" % ( variables_search['depth'],
        variables_search['nn_size'],
        variables_search['batch_size'],
        variables_search['reg_l1'],
        variables_search['reg_l2'],
        variables_search['dropout'],
        variables_search['use_hpc'],
        variables_search['retrieve'])

    molcas_info = """
  &molcas
-------------------------------------------------------
  Molcas:                   %-10s
  Molcas_nproc:             %-10s
  Molcas_mem:               %-10s
  Molcas_print:      	    %-10s
  Molcas_project:      	    %-10s
  Molcas_workdir:      	    %-10s
  Molcas_calcdir:           %-10s
  Tinker interface:         %-10s
  Omp_num_threads:          %-10s
  Keep tmp_molcas:          %-10s
  Track phase:              %-10s
  Job distribution:         %-10s
-------------------------------------------------------
""" % ( variables_molcas['molcas'],
        variables_molcas['molcas_nproc'],    
        variables_molcas['molcas_mem'],
        variables_molcas['molcas_print'],
        variables_molcas['molcas_project'],
        variables_molcas['molcas_workdir'],
        variables_molcas['molcas_calcdir'],
        variables_molcas['tinker'],
        variables_molcas['omp_num_threads'],
        variables_molcas['keep_tmp'],
        variables_molcas['track_phase'],
        variables_molcas['use_hpc'])

    bagel_info="""
  &bagel
-------------------------------------------------------
  BAGEL:                    %-10s
  BAGEL_nproc:              %-10s
  BAGEL_project:            %-10s
  BAGEL_workdir:            %-10s
  BAGEL_archive:            %-10s
  MPI:                      %-10s
  BLAS:                     %-10s
  LAPACK:                   %-10s
  BOOST:                    %-10s
  MKL:                      %-10s
  Architecture:             %-10s
  Omp_num_threads:          %-10s
  Keep tmp_bagel:           %-10s
  Job distribution:         %-10s
-------------------------------------------------------
""" % ( variables_bagel['bagel'],
        variables_bagel['bagel_nproc'],
        variables_bagel['bagel_project'],
        variables_bagel['bagel_workdir'],
        variables_bagel['bagel_archive'],
        variables_bagel['mpi'],
        variables_bagel['blas'],
        variables_bagel['lapack'],
        variables_bagel['boost'],
        variables_bagel['mkl'],
        variables_bagel['arch'],
        variables_bagel['omp_num_threads'],
        variables_bagel['keep_tmp'],
        variables_bagel['use_hpc'])

    info_method = {
    'nn'    :   nn_info,
    'molcas':   molcas_info,
    'mlctkr':   molcas_info,
    'bagel' :   bagel_info
    }

    ## unpack control variables
    jobtype = variables_all['control']['jobtype']
    qm      = variables_control['qm']
    abinit  = variables_control['abinit']

    info_jobtype = {
    'sp'         : control_info + molecule_info + info_method[qm],
    'md'         : control_info + molecule_info + md_info + info_method[qm],
    'hop'        : control_info + molecule_info + md_info,
    'hybrid'     : control_info + molecule_info + md_info + info_method[qm] + info_method[abinit] + hybrid_info,
    'adaptive'   : control_info + molecule_info + adaptive_info + md_info + info_method[qm] + info_method[abinit],
    'train'      : control_info + molecule_info + info_method[qm],
    'prediction' : control_info + molecule_info + info_method[qm],
    'predict'    : control_info + molecule_info + info_method[qm],
    'search'     : control_info + molecule_info + info_method[qm] + search_info,
    }

    log_info = info_jobtype[jobtype]

    return log_info

