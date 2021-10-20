import tensorflow as tf
import tensorflow.keras as ks


class InverseDistanceIndexed(ks.layers.Layer):
    """
    Compute inverse distances from coordinates.
    
    The index-list of atoms to compute distances from is added as a static non-trainable weight.
    This should be cleaner than always have to move the index within the model.
    """

    def __init__(self, invd_shape, **kwargs):
        """
        Init the layer. The index list is initialized to zero.

        Args:
            invd_shape (list): Shape of the index piar list without batch dimension (N,2).
            **kwargs.
            
        """
        super(InverseDistanceIndexed, self).__init__(**kwargs)
        self.invd_shape = invd_shape

        self.invd_list = self.add_weight('invd_list',
                                         shape=invd_shape,
                                         initializer=tf.keras.initializers.Zeros(),
                                         dtype='int64',
                                         trainable=False)

    def build(self, input_shape):
        """
        Build model. Index list is built in init.

        Args:
            input_shape (list): Input shape.

        """
        super(InverseDistanceIndexed, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Coordinate input as (batch,N,3).

        Returns:
            angs_rad (tf.tensor): Flatten list of angles from index.

        """
        cordbatch = inputs
        invdbatch = tf.repeat(ks.backend.expand_dims(self.invd_list, axis=0), ks.backend.shape(cordbatch)[0], axis=0)
        vcords1 = tf.gather(cordbatch, invdbatch[:, :, 0], axis=1, batch_dims=1)
        vcords2 = tf.gather(cordbatch, invdbatch[:, :, 1], axis=1, batch_dims=1)
        vec = vcords2 - vcords1
        norm_vec = ks.backend.sqrt(ks.backend.sum(vec * vec, axis=-1))
        invd_out = tf.math.divide_no_nan(tf.ones_like(norm_vec), norm_vec)
        return invd_out

    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus angle invd shape.

        """
        config = super(InverseDistanceIndexed, self).get_config()
        config.update({"invd_shape": self.invd_shape})
        return config


class Angles(ks.layers.Layer):
    """
    Compute angles from coordinates.
    
    The index-list of atoms to compute angles from is added as a static non-trainable weight.
    This should be cleaner than always have to move the index within the model.
    """

    def __init__(self, angle_shape, **kwargs):
        """
        Init the layer. The angle list is initialized to zero.

        Args:
            angle_shape (list): Shape of the angle list without batch dimension (N,3).
            **kwargs.
            
        """
        super(Angles, self).__init__(**kwargs)
        # self.angle_list = angle_list
        # self.angle_list_tf = tf.constant(np.array(angle_list))
        self.angle_list = self.add_weight('angle_list',
                                          shape=angle_shape,
                                          initializer=tf.keras.initializers.Zeros(),
                                          dtype='int64',
                                          trainable=False)
        self.angle_shape = angle_shape

    def build(self, input_shape):
        """
        Build model. Angle list is built in init.

        Args:
            input_shape (list): Input shape.

        """
        super(Angles, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Coordinate input as (batch,N,3).

        Returns:
            angs_rad (tf.tensor): Flatten list of angles from index.

        """
        cordbatch = inputs
        angbatch = tf.repeat(ks.backend.expand_dims(self.angle_list, axis=0), ks.backend.shape(cordbatch)[0], axis=0)
        vcords1 = tf.gather(cordbatch, angbatch[:, :, 1], axis=1, batch_dims=1)
        vcords2a = tf.gather(cordbatch, angbatch[:, :, 0], axis=1, batch_dims=1)
        vcords2b = tf.gather(cordbatch, angbatch[:, :, 2], axis=1, batch_dims=1)
        vec1 = vcords2a - vcords1
        vec2 = vcords2b - vcords1
        norm_vec1 = ks.backend.sqrt(ks.backend.sum(vec1 * vec1, axis=-1))
        norm_vec2 = ks.backend.sqrt(ks.backend.sum(vec2 * vec2, axis=-1))
        angle_cos = ks.backend.sum(vec1 * vec2, axis=-1) / norm_vec1 / norm_vec2
        angs_rad = tf.math.acos(angle_cos)
        return angs_rad

    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus angle index shape.

        """
        config = super(Angles, self).get_config()
        config.update({"angle_shape": self.angle_shape})
        return config


class Dihedral(ks.layers.Layer):
    """
    Compute dihedral angles from coordinates.
    
    The index-list of atoms to compute angles from is added as a static non-trainable weight.
    This should be cleaner than always have to move the index within the model.
    """

    def __init__(self, dihed_shape, **kwargs):
        """
        Init the layer. The angle list is initialized to zero.

        Args:
            angle_shape (list): Shape of the angle list without batch dimension of (N,4).
            **kwargs

        """
        super(Dihedral, self).__init__(**kwargs)
        # self.angle_list = angle_list
        # self.angle_list_tf = tf.constant(np.array(angle_list))
        self.dihed_list = self.add_weight('dihed_list',
                                          shape=dihed_shape,
                                          initializer=tf.keras.initializers.Zeros(),
                                          dtype='int64',
                                          trainable=False)
        self.dihed_shape = dihed_shape

    def build(self, input_shape):
        """
        Build model. Angle list is built in init.

        Args:
            input_shape (list): Input shape.

        """
        super(Dihedral, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass.

        Args:
            inputs (tf.tensor): Coordinates of shape (batch, N,3).

        Returns:
            angs_rad (tf.tensor): Dihydral angles from index list and coordinates of shape (batch,M).

        """
        # implementation from
        # https://en.wikipedia.org/wiki/Dihedral_angle
        cordbatch = inputs
        indexbatch = tf.repeat(ks.backend.expand_dims(self.dihed_list, axis=0), ks.backend.shape(cordbatch)[0], axis=0)
        p1 = tf.gather(cordbatch, indexbatch[:, :, 0], axis=1, batch_dims=1)
        p2 = tf.gather(cordbatch, indexbatch[:, :, 1], axis=1, batch_dims=1)
        p3 = tf.gather(cordbatch, indexbatch[:, :, 2], axis=1, batch_dims=1)
        p4 = tf.gather(cordbatch, indexbatch[:, :, 3], axis=1, batch_dims=1)
        b1 = p1 - p2
        b2 = p2 - p3
        b3 = p4 - p3
        arg1 = ks.backend.sum(b2 * tf.linalg.cross(tf.linalg.cross(b3, b2), tf.linalg.cross(b1, b2)), axis=-1)
        arg2 = ks.backend.sqrt(ks.backend.sum(b2 * b2, axis=-1)) * ks.backend.sum(
            tf.linalg.cross(b1, b2) * tf.linalg.cross(b3, b2), axis=-1)
        angs_rad = tf.math.atan2(arg1, arg2)
        return angs_rad

    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus angle index shape.

        """
        config = super(Dihedral, self).get_config()
        config.update({"dihed_shape": self.angle_shape})
        return config


class FeatureGeometric(ks.layers.Layer):
    """
    Feautre representation consisting of inverse distances, angles and dihedral angles.
    
    Uses InverseDistance, Angle, Dihydral layer definition if input index is not empty.
    
    """

    def __init__(self,
                 invd_shape=None,
                 angle_shape=None,
                 dihed_shape=None,
                 **kwargs):
        """
        Init of the layer.

        Args:
            invd_shape (list, optional): Index-Shape of atoms to calculate inverse distances. Defaults to None.
            angle_shape (list, optional): Index-Shape of atoms to calculate angles between. Defaults to None.
            dihed_shape (list, optional): Index-Shape of atoms to calculate dihed between. Defaults to None.
            **kwargs

        """
        super(FeatureGeometric, self).__init__(**kwargs)
        # Inverse distances are always taken all for the moment
        self.use_invdist = invd_shape is not None
        self.invd_shape = invd_shape
        self.use_bond_angles = angle_shape is not None
        self.angle_shape = angle_shape
        self.use_dihed_angles = dihed_shape is not None
        self.dihed_shape = dihed_shape

        if not self.use_invdist and not self.use_bond_angles and not self.use_dihed_angles:
            raise ValueError("Feature Layer: One geometric feature type must be defined or features = [].")

        if self.use_invdist:
            self.invd_layer = InverseDistanceIndexed(invd_shape)
        if self.use_bond_angles:
            self.ang_layer = Angles(angle_shape=angle_shape)
            self.concat_ang = ks.layers.Concatenate(axis=-1)
        if self.use_dihed_angles:
            self.dih_layer = Dihedral(dihed_shape=dihed_shape)
            self.concat_dih = ks.layers.Concatenate(axis=-1)
        self.flat_layer = ks.layers.Flatten(name='feat_flat')

    def build(self, input_shape):
        """
        Build model. Passes to base class.

        Args:
            input_shape (list): Input shape.

        """
        super(FeatureGeometric, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Forward pass of the layer. Call().

        Args:
            inputs (tf.tensor): Coordinates of shape (batch,N,3).

        Returns:
            out (tf.tensor): Feature description of shape (batch,M).

        """
        x = inputs

        feat = None
        if self.use_invdist:
            feat = self.invd_layer(x)
        if self.use_bond_angles:
            if not self.use_invdist:
                feat = self.ang_layer(x)
            else:
                angs = self.ang_layer(x)
                feat = self.concat_ang([feat, angs])
        if self.use_dihed_angles:
            if not self.use_invdist and not self.use_bond_angles:
                feat = self.dih_layer(x)
            else:
                dih = self.dih_layer(x)
                feat = self.concat_dih([feat, dih])

        feat_flat = self.flat_layer(feat)
        out = feat_flat
        return out

    def set_mol_index(self, invd_index, angle_index, dihed_index):
        """
        Set weights for atomic index for distance and angles.

        Args:
            invd_index (np.array): Index for inverse distances. Shape (N,2)
            angle_index (np.array): Index for angles. Shape (N,3).
            dihed_index (np.array):Index for dihed angles. Shape (N,4).

        """
        if self.use_invdist:
            self.invd_layer.set_weights([invd_index])
        if self.use_dihed_angles:
            self.dih_layer.set_weights([dihed_index])
        if self.use_bond_angles:
            self.ang_layer.set_weights([angle_index])

    def get_config(self):
        """
        Return config for layer.

        Returns:
            config (dict): Config from base class plus index info.

        """
        config = super(FeatureGeometric, self).get_config()
        config.update({"invd_shape": self.invd_shape,
                       "angle_shape": self.angle_shape,
                       "dihed_shape": self.dihed_shape
                       })
        return config

    def get_feature_type_segmentation(self):
        """
        Get the feature output segmentation length [invd,angle,dihys]

        Returns:
             feat_segments (list): Segmentation length
        """
        feat_segments = []
        if self.invd_shape is not None:
            feat_segments.append(self.invd_shape[0])
        if self.angle_shape is not None:
            feat_segments.append(self.angle_shape[0])
        if self.dihed_shape is not None:
            feat_segments.append(self.dihed_shape[0])
        return feat_segments


class InverseDistance(ks.layers.Layer):
    def __init__(self, **kwargs):
        super(InverseDistance, self).__init__(**kwargs)
        # self.dinv_mean = dinv_mean
        # self.dinv_std = dinv_std

    def build(self, input_shape):
        super(InverseDistance, self).build(input_shape)

    def call(self, inputs, **kwargs):
        coords = inputs  # (batch,N,3)
        # Compute square dinstance matrix
        ins_int = ks.backend.int_shape(coords)
        ins = ks.backend.shape(coords)
        a = ks.backend.expand_dims(coords, axis=1)
        b = ks.backend.expand_dims(coords, axis=2)
        c = b - a  # (batch,N,N,3)
        d = ks.backend.sum(ks.backend.square(c), axis=-1)  # squared distance without sqrt for derivative
        # Compute Mask for lower tri
        ind1 = ks.backend.expand_dims(ks.backend.arange(0, ins_int[1]), axis=1)
        ind2 = ks.backend.expand_dims(ks.backend.arange(0, ins_int[1]), axis=0)
        mask = ks.backend.less(ind1, ind2)
        mask = ks.backend.expand_dims(mask, axis=0)
        mask = ks.backend.tile(mask, (ins[0], 1, 1))  # (batch,N,N)
        # Apply Mask and reshape
        d = d[mask]
        d = ks.backend.reshape(d, (ins[0], ins_int[1] * (ins_int[1] - 1) // 2))  # Not pretty
        d = ks.backend.sqrt(d)  # Now the sqrt is okay
        out = 1 / d  # Now inverse should also be okay
        # out = (out - self.dinv_mean )/self.dinv_std #standardize with fixed values.
        return out
