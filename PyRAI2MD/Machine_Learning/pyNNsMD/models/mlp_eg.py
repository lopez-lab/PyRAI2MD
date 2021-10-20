"""
Tensorflow keras model definitions for energy and gradient.

There are two definitions: the subclassed EnergyGradientModel and a precomputed model to 
multiply with the feature derivative for training, which overwrites training/predict step.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from pyNNsMD.layers.features import FeatureGeometric
from pyNNsMD.layers.gradients import EmptyGradient
from pyNNsMD.layers.mlp import MLP
from pyNNsMD.layers.normalize import ConstLayerNormalization
from pyNNsMD.scaler.general import SegmentStandardScaler


class EnergyGradientModel(ks.Model):
    """
    Subclassed tf.keras.model for energy/gradient which outputs both energy and gradient from coordinates.
    
    This is not used for fitting, only for prediction as for fitting a feature-precomputed model is used instead.
    The model is supposed to be saved and exported for MD code.
    """

    def __init__(self,
                 states=1,
                 atoms=2,
                 invd_index=None,
                 angle_index=None,
                 dihed_index=None,
                 nn_size=100,
                 depth=3,
                 activ='selu',
                 use_reg_activ=None,
                 use_reg_weight=None,
                 use_reg_bias=None,
                 use_dropout=False,
                 dropout=0.01,
                 normalization_mode=1,
                 energy_only = False,
                 precomputed_features = False,
                 output_as_dict = False,
                 **kwargs):
        """
        Initialize Layer

        Args:
            states:
            atoms:
            invd_index:
            angle_index:
            dihed_index:
            nn_size:
            depth:
            activ:
            use_reg_activ:
            use_reg_weight:
            use_reg_bias:
            use_dropout:
            dropout:
            **kwargs:
        """

        super(EnergyGradientModel, self).__init__(**kwargs)
        self.in_invd_index = invd_index
        self.in_angle_index = angle_index
        self.in_dihed_index = dihed_index
        self.nn_size = nn_size
        self.depth = depth
        self.activ = activ
        self.use_reg_activ = use_reg_activ
        self.use_reg_weight = use_reg_weight
        self.use_reg_bias = use_reg_bias
        self.use_dropout = use_dropout
        self.dropout = dropout
        # Control the properties
        self.energy_only = energy_only
        self.output_as_dict = output_as_dict
        self.eg_atoms = int(atoms)
        self.eg_states = int(states)
        self.normalization_mode = normalization_mode

        out_dim = int(states)
        indim = int(atoms)

        # Allow for all distances, backward compatible
        if isinstance(invd_index, bool):
            if invd_index:
                invd_index = [[i, j] for i in range(0, int(atoms)) for j in range(0, i)]

        use_invd_index = len(invd_index) > 0 if isinstance(invd_index, list) or isinstance(invd_index,
                                                                                           np.ndarray) else False
        use_angle_index = len(angle_index) > 0 if isinstance(angle_index, list) or isinstance(angle_index,
                                                                                              np.ndarray) else False
        use_dihed_index = len(dihed_index) > 0 if isinstance(dihed_index, list) or isinstance(dihed_index,
                                                                                              np.ndarray) else False

        invd_index = np.array(invd_index, dtype=np.int64) if use_invd_index else None
        angle_index = np.array(angle_index, dtype=np.int64) if use_angle_index else None
        dihed_index = np.array(dihed_index, dtype=np.int64) if use_dihed_index else None

        invd_shape = invd_index.shape if use_invd_index else None
        angle_shape = angle_index.shape if use_angle_index else None
        dihed_shape = dihed_index.shape if use_dihed_index else None

        self.feat_layer = FeatureGeometric(invd_shape=invd_shape,
                                           angle_shape=angle_shape,
                                           dihed_shape=dihed_shape,
                                           name="feat_geo"
                                           )
        self.feat_layer.set_mol_index(invd_index, angle_index, dihed_index)

        self.std_layer = ConstLayerNormalization(axis=-1, name='feat_std')
        self.mlp_layer = MLP(nn_size,
                             dense_depth=depth,
                             dense_bias=True,
                             dense_bias_last=True,
                             dense_activ=activ,
                             dense_activ_last=activ,
                             dense_activity_regularizer=use_reg_activ,
                             dense_kernel_regularizer=use_reg_weight,
                             dense_bias_regularizer=use_reg_bias,
                             dropout_use=use_dropout,
                             dropout_dropout=dropout,
                             name='mlp'
                             )
        self.energy_layer = ks.layers.Dense(out_dim, name='energy', use_bias=True, activation='linear')
        self.force = EmptyGradient(mult_states=out_dim, atoms=indim,
                                   name='force')  # Will be differentiated in fit/predict/evaluate

        # Need to build model already to set std layer
        self.precomputed_features = False
        self.build((None, indim, 3))

        self.precomputed_features = precomputed_features

    def call(self, data, training=False, **kwargs):
        """
        Call the model output, forward pass.

        Args:
            data (tf.tensor): Coordinates.
            training (bool, optional): Training Mode. Defaults to False.

        Returns:
            y_pred (list): List of tf.tensor for predicted [energy,gradient]

        """
        # Unpack the data
        x = data
        y_pred = None
        if self.energy_only and not self.precomputed_features:
            feat_flat = self.feat_layer(x)
            feat_flat_std = self.std_layer(feat_flat)
            temp_hidden = self.mlp_layer(feat_flat_std, training=training)
            temp_e = self.energy_layer(temp_hidden)
            temp_g = self.force(x)
            y_pred = [temp_e, temp_g]
        elif not self.energy_only and not self.precomputed_features:
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                feat_flat = self.feat_layer(x)
                feat_flat_std = self.std_layer(feat_flat)
                temp_hidden = self.mlp_layer(feat_flat_std, training=training)
                temp_e = self.energy_layer(temp_hidden)
            temp_g = tape2.batch_jacobian(temp_e, x)
            _ = self.force(x)
            y_pred = [temp_e, temp_g]
        elif self.precomputed_features and not self.energy_only:
            x1 = x[0]
            x2 = x[1]
            with tf.GradientTape() as tape2:
                tape2.watch(x1)
                feat_flat_std = self.std_layer(x1)
                temp_hidden = self.mlp_layer(feat_flat_std, training=training)
                atpot = self.energy_layer(temp_hidden)
            grad = tape2.batch_jacobian(atpot, x1)
            grad = ks.backend.batch_dot(grad, x2, axes=(2, 1))
            y_pred = [atpot, grad]
        elif self.precomputed_features and self.energy_only:
            x1 = x[0]
            # x2 = x[1]
            feat_flat_std = self.std_layer(x1)
            temp_hidden = self.mlp_layer(feat_flat_std, training=training)
            temp_e = self.energy_layer(temp_hidden)
            temp_g = self.force(x1)
            y_pred = [temp_e, temp_g]

        if self.output_as_dict:
            out = {'energy': y_pred[0], 'force': y_pred[1]}
        else:
            out = y_pred
        return out

    @tf.function
    def predict_chunk_feature(self, tf_x):
        with tf.GradientTape() as tape2:
            tape2.watch(tf_x)
            feat_pred = self.feat_layer(tf_x, training=False)  # Forward pass
        grad = tape2.batch_jacobian(feat_pred, tf_x)
        return feat_pred, grad

    def precompute_feature_in_chunks(self, x, batch_size):
        np_x = []
        np_grad = []
        for j in range(int(np.ceil(len(x) / batch_size))):
            a = int(batch_size * j)
            b = int(batch_size * j + batch_size)
            tf_x = tf.convert_to_tensor(x[a:b], dtype=tf.float32)
            feat_pred, grad = self.predict_chunk_feature(tf_x)
            np_x.append(np.array(feat_pred.numpy()))
            np_grad.append(np.array(grad.numpy()))

        np_x = np.concatenate(np_x, axis=0)
        np_grad = np.concatenate(np_grad, axis=0)

        return np_x, np_grad

    def set_const_normalization_from_features(self, feat_x, normalization_mode=None):
        if normalization_mode is None:
            normalization_mode = self.normalization_mode
        else:
            self.normalization_mode = normalization_mode

        feat_x_mean, feat_x_std = self.get_layer('feat_std').get_weights()
        if normalization_mode == 1:
            feat_x_mean = np.mean(feat_x, axis=0, keepdims=True)
            feat_x_std = np.std(feat_x, axis=0, keepdims=True)
        elif normalization_mode == 2:
            seg_scaler = SegmentStandardScaler(self.get_layer('feat_geo').get_feature_type_segmentation())
            seg_scaler.fit(y=feat_x)
            feat_x_mean, feat_x_std = np.array(seg_scaler.get_params()["feat_mean"]), np.array(
                seg_scaler.get_params()["feat_std"])

        self.get_layer('feat_std').set_weights([feat_x_mean, feat_x_std])

        return [feat_x_mean, feat_x_std]

    def fit(self, **kwargs):

        if self.precomputed_features:
            self.set_const_normalization_from_features(kwargs['x'][0])

        return super(EnergyGradientModel, self).fit(**kwargs)

    def get_config(self):
        # conf = super(EnergyGradientModel, self).get_config()
        conf = {}
        conf.update({
            'atoms': self.eg_atoms,
            'states': self.eg_states,
            'invd_index': self.in_invd_index,
            'angle_index': self.in_angle_index,
            'dihed_index': self.in_dihed_index,
            'nn_size': self.nn_size,
            'depth': self.depth,
            'activ': self.activ,
            'use_reg_activ': self.use_reg_activ,
            'use_reg_weight': self.use_reg_weight,
            'use_reg_bias': self.use_reg_bias,
            'use_dropout': self.use_dropout,
            'dropout': self.dropout,
            'normalization_mode': self.normalization_mode,
            'energy_only': self.energy_only,
            'precomputed_features': self.precomputed_features,
            'output_as_dict': self.output_as_dict
        })
        return conf

    def save(self,filepath,**kwargs):
        # copy to new model
        self_conf = self.get_config()
        self_conf['precomputed_features'] = False
        copy_model = EnergyGradientModel(**self_conf)
        copy_model.set_weights(self.get_weights())
        # Make graph and test with training data
        copy_model.predict(np.ones((1,self.eg_atoms,3)))
        tf.keras.models.save_model(copy_model,filepath,**kwargs)