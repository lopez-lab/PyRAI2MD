"""
Default hyperparameters for MLP model for nac
"""

DEFAULT_HYPER_PARAM_NAC = {
    'general':
        {
            'model_type': 'mlp_nac',
            'main_dir': '',
            'model_dir': '',
            'info': '',
            'pyNN_version': "1.0.2"  # not used atm
        },
    'model':  # Model Parameters # fixed model, cannot be changed after init
        {
            'atoms': 2,
            'states': 1,  # (batch,states*(states-1)/2,atoms,3)
            'nnac'  : 1,
            'depth': 3,
            'activ': {'class_name': "leaky_softplus", "config": {'alpha': 0.03}},  # activation function,
            'nn_size': 100,
            # Regularization
            'dropout': 0.005,
            'use_dropout': False,
            'use_reg_activ': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            'use_reg_weight': None,
            'use_reg_bias': None,
            # features
            'invd_index': True,  # not used yet
            'angle_index': [],  # list-only of shape (N,3) angle: 0-1-2  or alpha(1->0,1->2)
            'dihed_index': [],  # list of dihedral angles (N,4) with index ijkl angle is between ijk and jkl
        },
    'training': {
        # Fit information
        'auto_scaling': {'x_mean': False, 'x_std': False, 'nac_std': True, 'nac_mean': False},
        # Scale nac und coordinates, can be also done in data preparation,
        'normalization_mode': 1,
        'learning_rate': 1e-3,
        'phase_less_loss': True,
        'initialize_weights': True,
        'val_disjoint': True,
        'val_split': 0.1,
        'epo': 3000,
        'pre_epo': 50,  # number of epochs without phaseless loss
        'epostep': 10,
        'batch_size': 64,
        # Callbacks
        'step_callback': {'use': False, 'epoch_step_reduction': [500, 1500, 500, 500],
                          'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6]},
        'linear_callback': {'use': False, 'learning_rate_start': 1e-3, 'learning_rate_stop': 1e-6, 'epomin': 100,
                            'epo': 1000},
        'early_callback': {'use': False, 'epomin': 5000, 'patience': 600, 'max_time': 600, 'delta_loss': 1e-5,
                           'loss_monitor': 'val_loss', 'factor_lr': 0.1, 'learning_rate_start': 1e-3,
                           'learning_rate_stop': 1e-6, 'epostep': 1},
        'exp_callback': {'use': False, 'factor_lr': 0.1, 'epomin': 100, 'learning_rate_start': 1e-3},
        },
    'retraining': {
        # Fit information
        'auto_scaling': {'x_mean': False, 'x_std': False, 'nac_std': True, 'nac_mean': False},
        # Scale nac und coordinates, can be also done in data preparation,
        'normalization_mode': 1,
        'learning_rate': 1e-3,
        'phase_less_loss': True,
        'initialize_weights': False,  # To take old weights
        'val_disjoint': True,
        'val_split': 0.1,
        'epo': 1000,
        'pre_epo': 50,  # number of epochs without phaseless loss
        'epostep': 10,
        'batch_size': 64,
        # Callbacks
        'step_callback': {'use': False, 'epoch_step_reduction': [500, 1500, 500, 500],
                          'learning_rate_step': [1e-3, 1e-4, 1e-5, 1e-6]},
        'linear_callback': {'use': False, 'learning_rate_start': 1e-3, 'learning_rate_stop': 1e-6, 'epomin': 100,
                            'epo': 1000},
        'early_callback': {'use': False, 'epomin': 5000, 'patience': 600, 'max_time': 600, 'delta_loss': 1e-5,
                           'loss_monitor': 'val_loss', 'factor_lr': 0.1, 'learning_rate_start': 1e-3,
                           'learning_rate_stop': 1e-6, 'epostep': 1},
        'exp_callback': {'use': False, 'factor_lr': 0.1, 'epomin': 100, 'learning_rate_start': 1e-3},
        },
    "predict":
        {
            'batch_size_predict': 265,
            'try_predict_hessian': False,  # not implemented yet
        },
    'plots':
        {
            'unit_nac': "1/A"
        }
}
