
DEFAULT_HYPER_PARAM_GRADS2 = {

    'general':
        {
            'model_type': 'mlp_g2',  # which model type to use
            'main_dir': '',  # not used atm
            'model_dir': '',  # not used atm
            'info': '',  # not used atm
            'pyNN_version': "1.0.2"  # not used atm
        },
    'model':  # Model Parameters   # fixes model, cannot be changed after init
        {
            'atoms': 2,  # number of atoms
            'states': 1,  # (batch,states) and (batch,states,atoms,3)
            'nn_size': 100,  # size of each layer
            'depth': 3,  # number of layers
            'activ': {'class_name': "leaky_softplus", "config": {'alpha': 0.03}},  # activation function
            # Regularozation
            'use_dropout': False,  # Whether to use dropout
            'dropout': 0.005,  # dropout values
            'use_reg_activ': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            'use_reg_weight': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            'use_reg_bias': None,  # {'class_name': 'L1', 'config': {'l1': 0.009999999776482582}}
            # Features
            'invd_index': True,  # not used yet
            'angle_index': [],  # list-only of shape (N,3) angle: 0-1-2  or alpha(1->0,1->2)
            'dihed_index': [],  # list of dihedral angles with index ijkl angle is between ijk and jkl
        },
    'training':
        {
            # can be changed after model created
            'auto_scaling': {'x_mean': True, 'x_std': True, 'gradient_std': True, 'gradient_mean': True},
            # Scale energy und coordinates, can be also done in data preparation
            'normalization_mode': 1,  # Normalization False/0 for no normalization/unity mulitplication
            'loss_weights': [1, 10],  # weights between energy and gradients
            'learning_rate': 1e-3,  # learning rate, can be modified by callbacks
            'initialize_weights': True,
            # Whether to reset the weights before fit, used for retraining, transfer learning
            'val_disjoint': True,  # Should be removed as always a disjoint validation split per instance is favourable
            'val_split': 0.1,
            'epo': 3000,  # total epochs
            'batch_size': 64,  # batch size
            'epostep': 10,  # steps of epochs for validation, also steps for changing callbacks
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
    'retraining':
        {
            # can be changed after model created
            'auto_scaling': {'x_mean': True, 'x_std': True, 'gradient_std': True, 'gradient_mean': False},
            # Scale energy und coordinates, can be also done in data preparation
            'normalization_mode': 1,  # Normalization False/0 for no normalization/unity mulitplication
            'learning_rate': 1e-3,  # learning rate, can be modified by callbacks
            'initialize_weights': False,
            # Whether to reset the weights before fit, used for retraining, transfer learning
            'val_disjoint': True,  # Should be removed as always a disjoint validation split per instance is favourable
            'val_split': 0.1,
            'epo': 1000,  # total epochs
            'batch_size': 64,  # batch size
            'epostep': 10,  # steps of epochs for validation, also steps for changing callbacks
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
    'predict':
        {
            'batch_size_predict': 265,
            'try_predict_hessian': False,  # not implemented yet
        },
    'plots':
        {
            'unit_gradient': "eV/A"
        }
}
