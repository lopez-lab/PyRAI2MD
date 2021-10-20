#####################################################
#
# PyRAI2MD 2 module for NNsForMD hyperparameter
#
# Author Jingbai Li
# Sep 22 2021
#
######################################################

import numpy as np

def SetHyperEG(hyp, unit, info):
    """ Generating hyperparameter dict for energy+gradient NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup regularization dict
    for penalty in ['use_reg_activ', 'use_reg_weight', 'use_reg_bias']:
        penalty_key = '%s_dict' % (penalty)                
        if   hyp[penalty] == 'l1':
            hyp[penalty_key] = {'class_name' : 'l1', 'config' : {'l1': hyp['reg_l1']}}
        elif hyp[penalty] == 'l2':
            hyp[penalty_key] = {'class_name' : 'l2', 'config' : {'l2': hyp['reg_l2']}}
        elif hyp[penalty] == 'l1_l2':
            hyp[penalty_key] = {'class_name' : 'l1_l2', 'config' : {'l1': hyp['reg_l1'], 'l2': hyp['reg_l2']}}
        else:
            hyp[penalty_key] = None

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = ['eV','eV/A']
    else:
        hyp['unit'] = ['Eh','Eh/Bohr']

    ## setup hypers
    hyp_dict = {
        'general'                      : {
            'model_type'               : hyp['model_type'],
        },
        'model'                        : {
            'atoms'                    : info['natom'],
            'states'                   : info['nstate'], 
            'nn_size'                  : hyp['nn_size'],
            'depth'                    : hyp['depth'],
            'activ'                    : {
                'class_name'           : hyp['activ'],
                'config'               : {
                    'alpha'            : hyp['activ_alpha']
                 }
            },
            'use_dropout'              : hyp['use_dropout'],
            'dropout'                  : hyp['dropout'],
            'use_reg_activ'            : hyp['use_reg_activ_dict'],
            'use_reg_weight'           : hyp['use_reg_weight_dict'],
            'use_reg_bias'             : hyp['use_reg_bias_dict'],
            'invd_index'               : (np.array(hyp['invd_index']) - 1).tolist() if len(hyp['invd_index']) > 0 else True,
            'angle_index'              : (np.array(hyp['angle_index']) - 1).tolist(),
            'dihed_index'              : (np.array(hyp['dihed_index']) - 1).tolist(),
        },
       	'training'                     : {
            'auto_scaling'             : {
                'x_mean'               : hyp['scale_x_mean'],
                'x_std'                : hyp['scale_x_std'],
                'energy_mean'          : hyp['scale_y_mean'],
                'energy_std'           : hyp['scale_y_std'],
            },
            'normalization_mode'       : hyp['normalization_mode'],
            'loss_weights'             : hyp['loss_weights'],
            'learning_rate'            : hyp['learning_rate'],
            'initialize_weights'       : hyp['initialize_weights'],
            'val_disjoint'             : hyp['val_disjoint'],
            'val_split'                : hyp['val_split'],
            'epo'                      : hyp['epo'],
            'batch_size'               : hyp['batch_size'],
            'epostep'                  : hyp['epostep'],
            'step_callback'            : {
                'use'                  : hyp['use_step_callback'],
                'epoch_step_reduction' : hyp['epoch_step_reduction'],
                'learning_rate_step'   : hyp['learning_rate_step'],
            },
            'linear_callback'          : {
                'use'                  : hyp['use_linear_callback'],
                'learning_rate_start'  : hyp['learning_rate_start'],
                'learning_rate_stop'   : hyp['learning_rate_stop' ],
                'epomin'               : hyp['epomin'],
            },
       	    'early_callback'           : {
                'use'                  : hyp['use_early_callback'],
                'epomin'               : hyp['epomin'],
                'patience'             : hyp['patience'],
                'max_time'             : hyp['max_time'],
                'delta_loss'           : hyp['delta_loss'],
                'loss_monitor'         : hyp['loss_monitor'],
                'factor_lr'            : hyp['factor_lr'],
                'learning_rate_start'  : hyp['learning_rate_start'],
                'learning_rate_stop'   : hyp['learning_rate_stop' ],
            },
            'exp_callback'             : {
                'use'                  : hyp['use_exp_callback'],
                'epomin'               : hyp['epomin'],
                'factor_lr'            : hyp['factor_lr'],
            },
        },
       	'plots'                        : {
            'unit_energy'              : hyp['unit'][0],
            'unit_gradient'            : hyp['unit'][1],
       	},
    }

    hyp_dict['retraining'] = hyp_dict['training']

    return hyp_dict

def SetHyperNAC(hyp, unit, info):
    """ Generating hyperparameter dict for nonadiabatic coupling NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup regularization dict
    for penalty in ['use_reg_activ','use_reg_weight','use_reg_bias']:
        penalty_key='%s_dict' % (penalty)                
        if   hyp[penalty] == 'l1':
            hyp[penalty_key] = {'class_name' : 'l1', 'config' : {'l1': hyp['reg_l1']}}
        elif hyp[penalty] == 'l2':
            hyp[penalty_key] = {'class_name' : 'l2', 'config' : {'l2': hyp['reg_l2']}}
        elif hyp[penalty] == 'l1_l2':
            hyp[penalty_key] = {'class_name' : 'l1_l2', 'config' : {'l1': hyp['reg_l1'], 'l2': hyp['reg_l2']}}
        else:
            hyp[penalty_key] = None

    ## setup unit scheme
    if unit == 'si':
        hyp['unit'] = 'eV/A'
    elif unit == 'au':
        hyp['unit'] = 'Eh/A'
    elif unit == 'eha':
        hyp['unit']  = 'Eh/Bohr'
    else:
        hyp['unit']  = 'eV/A'

    ## setup hypers
    hyp_dict = {
        'general'                      : {
            'model_type'               : hyp['model_type'],
        },
        'model'                        : {
            'atoms'                    : info['natom'],
            'states'                   : info['nstate'],
            'nnac'                     : info['nnac'], 
            'nn_size'                  : hyp['nn_size'],
            'depth'                    : hyp['depth'],
            'activ'                    : {
                'class_name'           : hyp['activ'],
                'config'               : {
                    'alpha'            : hyp['activ_alpha'],
                },
            },
            'use_dropout'              : hyp['use_dropout'],
            'dropout'                  : hyp['dropout'],
            'use_reg_activ'            : hyp['use_reg_activ_dict'],
            'use_reg_weight'           : hyp['use_reg_weight_dict'], 
            'use_reg_bias'             : hyp['use_reg_bias_dict'],
            'invd_index'               : (np.array(hyp['invd_index']) - 1).tolist() if len(hyp['invd_index']) > 0 else True,
            'angle_index'              : (np.array(hyp['angle_index']) - 1).tolist(),
            'dihed_index'              : (np.array(hyp['dihed_index']) - 1).tolist(),
            },
        'training'                     : {
            'auto_scaling'             : {
                'x_mean'               : hyp['scale_x_mean'],
                'x_std'                : hyp['scale_x_std'],
                'nac_mean'             : hyp['scale_y_mean'],
                'nac_std'              : hyp['scale_y_std'],
            },
            'normalization_mode'       : hyp['normalization_mode'],
            'learning_rate'            : hyp['learning_rate'],
            'phase_less_loss'          : hyp['phase_less_loss'],
            'initialize_weights'       : hyp['initialize_weights'],
            'val_disjoint'             : hyp['val_disjoint'],
            'val_split'                : hyp['val_split'],
            'epo'                      : hyp['epo'],
            'pre_epo'                  : hyp['pre_epo'],
            'batch_size'               : hyp['batch_size'],
            'epostep'                  : hyp['epostep'],
            'step_callback'            : {
                'use'                  : hyp['use_step_callback'],
                'epoch_step_reduction' : hyp['epoch_step_reduction'],
                'learning_rate_step'   : hyp['learning_rate_step'],
            },
            'linear_callback'          : {
                'use'                  : hyp['use_linear_callback'],
                'learning_rate_start'  : hyp['learning_rate_start'],
                'learning_rate_stop'   : hyp['learning_rate_stop' ],
                'epomin'               : hyp['epomin'],
            },
       	    'early_callback'           : {
                'use'                  : hyp['use_early_callback'],
                'epomin'               : hyp['epomin'],
                'patience'             : hyp['patience'],
                'max_time'             : hyp['max_time'],
                'delta_loss'           : hyp['delta_loss'],
                'loss_monitor'         : hyp['loss_monitor'],
                'factor_lr'            : hyp['factor_lr'],
                'learning_rate_start'  : hyp['learning_rate_start'],
                'learning_rate_stop'   : hyp['learning_rate_stop' ],
            },
       	    'exp_callback'             : {
                'use'                  : hyp['use_exp_callback'],
                'epomin'               : hyp['epomin'],
                'factor_lr'            : hyp['factor_lr'],
            },
       	},
        'plots'                        : {
            'unit_nac'                 : hyp['unit'],
       	},
    }

    hyp_dict['retraining'] = hyp_dict['training']

    return hyp_dict

def SetHyperSOC(hyp, unit, info):
    """ Generating hyperparameter dict for spin-orbit couplig  NN

        Parameters:          Type:
            hyp              dict     hyperparameter input
            unit             str      unit scheme
            info             dict     training data information

        Return:              Type:
            hyp_ditc         dict     hyperparameter dict for NN

    """

    ## setup regularization dict
    for penalty in ['use_reg_activ','use_reg_weight','use_reg_bias']:
        penalty_key='%s_dict' % (penalty)                
        if   hyp[penalty] == 'l1':
            hyp[penalty_key] = {'class_name' : 'l1', 'config' : {'l1': hyp['reg_l1']}}
        elif hyp[penalty] == 'l2':
            hyp[penalty_key] = {'class_name' : 'l2', 'config' : {'l2': hyp['reg_l2']}}
        elif hyp[penalty] == 'l1_l2':
            hyp[penalty_key] = {'class_name' : 'l1_l2', 'config' : {'l1': hyp['reg_l1'], 'l2': hyp['reg_l2']}}
        else:
            hyp[penalty_key] = None

    ## setup unit scheme
    if unit == 'si':
        hyp['unit']  = 'cm-1'
    else:
        hyp['unit']  = 'cm-1'

    ## setup hypers
    hyp_dict = {
        'general'                      : {
            'model_type'               : hyp['model_type'],
        },
        'model'                        : {
            'atoms'                    : info['natom'],
            'states'                   : info['nsoc'], 
            'nn_size'                  : hyp['nn_size'],
            'depth'                    : hyp['depth'],
            'activ'                    : {
                'class_name'           : hyp['activ'],
                'config'               : {
                    'alpha'            : hyp['activ_alpha']
                 }
            },
            'use_dropout'              : hyp['use_dropout'],
            'dropout'                  : hyp['dropout'],
            'use_reg_activ'            : hyp['use_reg_activ_dict'],
            'use_reg_weight'           : hyp['use_reg_weight_dict'],
            'use_reg_bias'             : hyp['use_reg_bias_dict'],
            'invd_index'               : (np.array(hyp['invd_index']) - 1).tolist() if len(hyp['invd_index']) > 0 else True,
            'angle_index'              : (np.array(hyp['angle_index']) - 1).tolist(),
            'dihed_index'              : (np.array(hyp['dihed_index']) - 1).tolist(),
        },
       	'training'                     : {
            'auto_scaling'             : {
                'x_mean'               : hyp['scale_x_mean'],
                'x_std'                : hyp['scale_x_std'],
                'energy_mean'          : hyp['scale_y_mean'],
                'energy_std'           : hyp['scale_y_std'],
            },
            'normalization_mode'       : hyp['normalization_mode'],
            'learning_rate'            : hyp['learning_rate'],
            'initialize_weights'       : hyp['initialize_weights'],
            'val_disjoint'             : hyp['val_disjoint'],
            'val_split'                : hyp['val_split'],
            'epo'                      : hyp['epo'],
            'batch_size'               : hyp['batch_size'],
            'epostep'                  : hyp['epostep'],
            'step_callback'            : {
                'use'                  : hyp['use_step_callback'],
                'epoch_step_reduction' : hyp['epoch_step_reduction'],
                'learning_rate_step'   : hyp['learning_rate_step'],
            },
            'linear_callback'          : {
                'use'                  : hyp['use_linear_callback'],
                'learning_rate_start'  : hyp['learning_rate_start'],
                'learning_rate_stop'   : hyp['learning_rate_stop' ],
                'epomin'               : hyp['epomin'],
            },
       	    'early_callback'           : {
                'use'                  : hyp['use_early_callback'],
                'epomin'               : hyp['epomin'],
                'patience'             : hyp['patience'],
                'max_time'             : hyp['max_time'],
                'delta_loss'           : hyp['delta_loss'],
                'loss_monitor'         : hyp['loss_monitor'],
                'factor_lr'            : hyp['factor_lr'],
                'learning_rate_start'  : hyp['learning_rate_start'],
                'learning_rate_stop'   : hyp['learning_rate_stop' ],
            },
            'exp_callback'             : {
                'use'                  : hyp['use_exp_callback'],
                'epomin'               : hyp['epomin'],
                'factor_lr'            : hyp['factor_lr'],
            },
        },
       	'plots'                        : {
            'unit_energy'              : hyp['unit'],
       	},
    }

    hyp_dict['retraining'] = hyp_dict['training']

    return hyp_dict
