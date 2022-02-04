import tensorflow as tf
import tensorflow.keras as ks


class EmptyGradient(ks.layers.Layer):
    """
    Layer to generate empty gradient output.
    """

    def __init__(self, mult_states=1, atoms=1, **kwargs):
        """
        Initialize empty gradient layer.

        Args:
            mult_states (int): Number of output states.
            atoms (int): Number of atoms.
            **kwargs
        """
        super(EmptyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms
        self.out_shape = tf.constant([self.mult_states,self.atoms,3],dtype=tf.int32)

    def build(self, input_shape):
        """Build layer."""
        super(EmptyGradient, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Generate any empty gradient placehodler.

        Args:
            inputs (tf.tensor): Energy tensor.
            **kwargs:

        Returns:

        """
        pot = inputs
        batch_shape = tf.expand_dims(tf.shape(pot)[0],axis=0)
        out_shape = tf.concat((batch_shape, tf.cast(self.out_shape,dtype=batch_shape.dtype)),axis=0)
        out = tf.zeros(out_shape)
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(EmptyGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class PropagateEnergyGradient(ks.layers.Layer):
    """
    Layer to propagate the gradients with precomputed layers.
    """

    def __init__(self, mult_states=1, **kwargs):
        """
        Initialize layer.

        Args:
            mult_states (int): Number of states
            **kwargs
        """
        super(PropagateEnergyGradient, self).__init__(**kwargs)
        self.mult_states = mult_states

    def build(self, input_shape):
        """Build layer."""
        super(PropagateEnergyGradient, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Propagate gradients

        Args:
            inputs: [grads, grads2]
            - grads (tf.tensor): Gradient for NN of shape (batch, states, features)
            - grads2 (tf.tensor): Gradients of static features. (batch, features, atoms, 3)
            **kwargs:

        Returns:
            out (tf.tensor): Gradients with respect to coordinates.
        """
        grads, grads2 = inputs
        out = ks.backend.batch_dot(grads, grads2, axes=(2, 1))
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(PropagateEnergyGradient, self).get_config()
        config.update({"mult_states": self.mult_states})
        return config


class PropagateNACGradient(ks.layers.Layer):
    """
    Propagate partial gradients for virtual NAC potentials.
    """

    def __init__(self, mult_states=1, atoms=1, **kwargs):
        """
        Initialize layer.

        Args:
            mult_states (int): number of states
            atoms (int): number of atoms
            **kwargs:
        """
        super(PropagateNACGradient, self).__init__(**kwargs)
        self.mult_states = mult_states
        self.atoms = atoms

    def build(self, input_shape):
        """Build layer."""
        super(PropagateNACGradient, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Propagate gradients for virtual NACs.

        Args:
            inputs: [grads, grads2]
            - grads (tf.tensor): Gradient for NN of shape (batch, states, atoms, features)
            - grads2 (tf.tensor): Gradients of static features. (batch, features, atoms, 3)
            **kwargs:

        Returns:
            out (tf.tensor): Gradients with respect to coordinates.
        """
        grads, grads2 = inputs
        out = ks.backend.batch_dot(grads, grads2, axes=(3, 1))
        out = ks.backend.concatenate([ks.backend.expand_dims(out[:, :, i, i, :], axis=2) for i in range(self.atoms)],
                                     axis=2)
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(PropagateNACGradient, self).get_config()
        config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
        return config


class PropagateNACGradient2(ks.layers.Layer):
    """
    Layer to propagate direct gradient predictions for NACs.
    """

    def __init__(self, axis=(2, 1), **kwargs):
        """
        Initialize layer

        Args:
            axis (tuple): Which axis the batch-dot is done. Default is (2,1)
            **kwargs:
        """
        super(PropagateNACGradient2, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        """Build layer."""
        super(PropagateNACGradient2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Simple batch-dot for partial gradients.

        Args:
            inputs: [grads, grads2]
            - grads (tf.tensor): Gradient for NN.
            - grads2 (tf.tensor): Gradients of static features.
            **kwargs:

        Returns:

        """
        grads, grads2 = inputs
        out = ks.backend.batch_dot(grads, grads2, axes=self.axis)
        return out

    def get_config(self):
        """Update config for layer."""
        config = super(PropagateNACGradient2, self).get_config()
        config.update({"axis": self.axis})
        return config

# class EnergyGradient(ks.layers.Layer):
#     """
#     Layer to calculate Gradient for NN energy output. Not used anymore.
#     """
#
#     def __init__(self, mult_states=1, **kwargs):
#         """
#         Initialize layer.
#
#         Args:
#             mult_states: Number of states.
#             **kwargs:
#         """
#         super(EnergyGradient, self).__init__(**kwargs)
#         self.mult_states = mult_states
#
#     def build(self, input_shape):
#         """Build layer."""
#         super(EnergyGradient, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         """
#         Calculate gradients.
#
#         Args:
#             inputs(list): [energy,coords]
#             - energy(tf.tensor): Energy output of shape (batch, states)
#             - coords (tf.tensor): Coordinates of shape (batch,atoms,3)
#             **kwargs:
#
#         Returns:
#             out (tf.tensor): forces
#         """
#         energy, coords = inputs
#         out = [ks.backend.expand_dims(ks.backend.gradients(energy[:, i], coords)[0], axis=1) for i in
#                range(self.mult_states)]
#         out = ks.backend.concatenate(out, axis=1)
#         return out
#
#     def get_config(self):
#         """Update config for layer."""
#         config = super(EnergyGradient, self).get_config()
#         config.update({"mult_states": self.mult_states})
#         return config


# class NACGradient(ks.layers.Layer):
#     """
#     Layer to calculate Gradient for NAC output. Not used anymore.
#     """
#
#     def __init__(self, mult_states=1, atoms=1, **kwargs):
#         """
#         Initialize layer.
#
#         Args:
#             mult_states: number of states
#             atoms: number of atoms
#             **kwargs:
#         """
#         super(NACGradient, self).__init__(**kwargs)
#         self.mult_states = mult_states
#         self.atoms = atoms
#
#     def build(self, input_shape):
#         """Build layer."""
#         super(NACGradient, self).build(input_shape)
#
#     def call(self, inputs, **kwargs):
#         """
#         Compute gradient for NACs.
#
#         Args:
#             inputs (list): [energy, coords]
#
#             - energy (tf.tensor): Virtual potentials for NAC. Shape (batch, states*atoms)
#             - coords (tf.tensor): Coordinates of shape (batch, atoms, 3)
#
#         Returns:
#             out (tf.tensor): NAC tensor of shape (batch, states, atoms, 3)
#         """
#         energy, coords = inputs
#         out = ks.backend.concatenate(
#             [ks.backend.expand_dims(ks.backend.gradients(energy[:, i], coords)[0], axis=1) for i in
#              range(self.mult_states * self.atoms)], axis=1)
#         out = ks.backend.reshape(out, (ks.backend.shape(coords)[0], self.mult_states, self.atoms, self.atoms, 3))
#         out = ks.backend.concatenate([ks.backend.expand_dims(out[:, :, i, i, :], axis=2) for i in range(self.atoms)],
#                                      axis=2)
#         return out
#
#     def get_config(self):
#         """Update config for layer."""
#         config = super(NACGradient, self).get_config()
#         config.update({"mult_states": self.mult_states, 'atoms': self.atoms})
#         return config
