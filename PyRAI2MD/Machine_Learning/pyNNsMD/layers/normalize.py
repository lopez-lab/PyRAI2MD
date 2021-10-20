import tensorflow as tf
import tensorflow.keras as ks
import numpy as np

class ConstLayerNormalization(ks.layers.Layer):
    """
    Layer normalization with constant scaler of input.

    Note that this sould be replaced with keras normalization layer where trainable could be altered.
    The standardization is done via 'std' and 'mean' tf.variable and uses not very flexible broadcasting.
    """

    def __init__(self, axis=-1, **kwargs):
        """
        Init the layer.

        Args:
            axis (int,list, optional): Which axis match the input on build. Defaults to -1.
            **kwargs

        """
        super(ConstLayerNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.wmean = None
        self.wstd = None

    def build(self, input_shape):
        """
        Build the layer.

        Args:
            input_shape (list): Shape of Input.

        Raises:
            TypeError: Axis argument is not valud.

        """
        super(ConstLayerNormalization, self).build(input_shape)
        outshape = [1] * len(input_shape)
        if isinstance(self.axis, int):
            outshape[self.axis] = input_shape[self.axis]
        elif isinstance(self.axis, list) or isinstance(self.axis, tuple):
            for i in self.axis:
                outshape[i] = input_shape[i]
        else:
            raise TypeError("Invalid axis argument")
        self.wmean = self.add_weight(
            'const_norm_mean',
            shape=outshape,
            initializer=tf.keras.initializers.Zeros(),
            dtype=self.dtype,
            trainable=False)
        self.wstd = self.add_weight(
            'const_norm_std',
            shape=outshape,
            initializer=tf.keras.initializers.Ones(),
            dtype=self.dtype,
            trainable=False)

    def call(self, inputs, **kwargs):
        """
        Forward pass of the layer. Call().

        Args:
            **kwargs:
            inputs (tf.tensor): Tensor to scale.

        Returns:
            out (tf.tensor): (inputs-mean)/std

        """
        out = (inputs - self.wmean) / (self.wstd + tf.keras.backend.epsilon())
        return out

    def get_config(self):
        """
        Config for the layer.

        Returns:
            config (dict): super.config with updated axis parameter.

        """
        config = super(ConstLayerNormalization, self).get_config()
        config.update({"axs": self.axis})
        return config

    def compute_const_normalization(self,feat_x):
        """
        Calculate and set the constant normalization of this layer.

        Args:
            feat (np.array): features of shape (batch,N)

        Returns:

        """
        feat_x_mean = np.mean(feat_x, axis=0, keepdims=True)
        feat_x_std = np.std(feat_x, axis=0, keepdims=True)
        self.set_weights([feat_x_mean, feat_x_std])
        return feat_x_mean,feat_x_std