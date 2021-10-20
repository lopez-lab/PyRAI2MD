# import tensorflow as tf
import tensorflow.keras as ks

from pyNNsMD.utils.activ import leaky_softplus, shifted_softplus


class MLP(ks.layers.Layer):
    """
    Multilayer perceptron that consist of N dense keras layers.

    Last layer can be modified sperately. Hidden layers are all the same.
    """

    def __init__(self,
                 dense_units,
                 dense_depth=1,
                 dense_bias=True,
                 dense_bias_last=True,
                 dense_activ=None,
                 dense_activ_last=None,
                 dense_activity_regularizer=None,
                 dense_kernel_regularizer=None,
                 dense_bias_regularizer=None,
                 dropout_use=False,
                 dropout_dropout=0,
                 **kwargs):
        """
        Init MLP as for dense.

        Args:
            dense_units (int): Size of hidden layers.
            dense_depth (int, optional): Number of hidden layers. Defaults to 1.
            dense_bias (bool, optional): Use bias for hidden layers. Defaults to True.
            dense_bias_last (bool, optional): Bias for last layer. Defaults to True.
            dense_activ (str, optional): Activity identifier. Defaults to None.
            dense_activ_last (str, optional): Activity identifier for last layer. Defaults to None.
            dense_activity_regularizer (str, optional): Activity regularizer identifier. Defaults to None.
            dense_kernel_regularizer (str, optional): Kernel regularizer identifier. Defaults to None.
            dense_bias_regularizer (str, optional): Bias regularizer identifier. Defaults to None.
            dropout_use (bool, optional): Use dropout. Defaults to False.
            dropout_dropout (float, optional): Fraction of dropout. Defaults to 0.
            **kwargs

        """
        super(MLP, self).__init__(**kwargs)
        self.dense_units = dense_units
        self.dense_depth = dense_depth
        self.dense_bias = dense_bias
        self.dense_bias_last = dense_bias_last
        self.dense_activ_serialize = dense_activ
        self.dense_activ = ks.activations.deserialize(dense_activ, custom_objects={'leaky_softplus': leaky_softplus,
                                                                                   'shifted_softplus': shifted_softplus
                                                                                   })
        self.dense_activ_last_serialize = dense_activ_last
        self.dense_activ_last = ks.activations.deserialize(dense_activ_last,
                                                           custom_objects={'leaky_softplus': leaky_softplus,
                                                                           'shifted_softplus': shifted_softplus})
        self.dense_activity_regularizer = ks.regularizers.get(dense_activity_regularizer)
        self.dense_kernel_regularizer = ks.regularizers.get(dense_kernel_regularizer)
        self.dense_bias_regularizer = ks.regularizers.get(dense_bias_regularizer)
        self.dropout_use = dropout_use
        self.dropout_dropout = dropout_dropout

        self.mlp_dense_activ = [ks.layers.Dense(
            self.dense_units,
            use_bias=self.dense_bias,
            activation=self.dense_activ,
            name=self.name + '_dense_' + str(i),
            activity_regularizer=self.dense_activity_regularizer,
            kernel_regularizer=self.dense_kernel_regularizer,
            bias_regularizer=self.dense_bias_regularizer
        ) for i in range(self.dense_depth - 1)]
        self.mlp_dense_last = ks.layers.Dense(
            self.dense_units,
            use_bias=self.dense_bias_last,
            activation=self.dense_activ_last,
            name=self.name + '_last',
            activity_regularizer=self.dense_activity_regularizer,
            kernel_regularizer=self.dense_kernel_regularizer,
            bias_regularizer=self.dense_bias_regularizer
        )
        if self.dropout_use:
            self.mlp_dropout = ks.layers.Dropout(self.dropout_dropout, name=self.name + '_dropout')

    def build(self, input_shape):
        """
        Build layer.

        Args:
            input_shape (list): Input shape.

        """
        super(MLP, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Input tensor of shape (...,N).
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            out (tf.tensor): Last activity.

        """
        x = inputs
        for i in range(self.dense_depth - 1):
            x = self.mlp_dense_activ[i](x)
            if self.dropout_use:
                x = self.mlp_dropout(x, training=training)
        x = self.mlp_dense_last(x)
        out = x
        return out

    def get_config(self):
        """
        Update config.

        Returns:
            config (dict): Base class config plus MLP info.

        """
        config = super(MLP, self).get_config()
        config.update({"dense_units": self.dense_units,
                       'dense_depth': self.dense_depth,
                       'dense_bias': self.dense_bias,
                       'dense_bias_last': self.dense_bias_last,
                       'dense_activ': self.dense_activ_serialize,
                       'dense_activ_last': self.dense_activ_last_serialize,
                       'dense_activity_regularizer': ks.regularizers.serialize(self.dense_activity_regularizer),
                       'dense_kernel_regularizer': ks.regularizers.serialize(self.dense_kernel_regularizer),
                       'dense_bias_regularizer': ks.regularizers.serialize(self.dense_bias_regularizer),
                       'dropout_use': self.dropout_use,
                       'dropout_dropout': self.dropout_dropout
                       })
        return config
