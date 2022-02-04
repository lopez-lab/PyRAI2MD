"""
Scaling of in and output
"""

import json

import numpy as np


class EnergyStandardScaler:
    def __init__(self):
        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.energy_mean = np.zeros((1, 1))
        self.energy_std = np.ones((1, 1))

        self._encountered_y_shape = None
        self._encountered_y_std = None

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            y_res = (y - self.energy_mean) / self.energy_std
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        energy = y
        x_res = x
        if y is not None:
            energy = y * self.energy_std + self.energy_mean
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        return x_res, energy

    def fit(self, x=None, y=None, auto_scale=None):
        if auto_scale is None:
            auto_scale = {'x_mean': True, 'x_std': True, 'energy_std': True, 'energy_mean': True}

        npeps = np.finfo(float).eps
        if auto_scale['x_mean']:
            self.x_mean = np.mean(x)
        if auto_scale['x_std']:
            self.x_std = np.std(x) + npeps
        if auto_scale['energy_mean']:
            self.energy_mean = np.mean(y, axis=0, keepdims=True)
        if auto_scale['energy_std']:
            self.energy_std = np.std(y, axis=0, keepdims=True) + npeps

        self._encountered_y_shape = np.array(y.shape)
        self._encountered_y_std = np.std(y, axis=0)

    def fit_transform(self, x=None, y=None, auto_scale=None):
        self.fit(x=x,y=y,auto_scale=auto_scale)
        return self.transform(x=x,y=y)

    def save(self, filepath):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'energy_mean': self.energy_mean.tolist(),
                   'energy_std': self.energy_std.tolist()
                   }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)

        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])

    def get_params(self):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'energy_mean': self.energy_mean.tolist(),
                   'energy_std': self.energy_std.tolist(),
                   }
        return outdict

    def set_params(self, indict):
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])

    def print_params_info(self):
        print("Info: Total-Data energy std", self._encountered_y_shape, ":", self._encountered_y_std)
        print("Info: Using energy-std", self.energy_std.shape, ":", self.energy_std)
        print("Info: Using energy-mean", self.energy_mean.shape, ":", self.energy_mean)
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)


class EnergyGradientStandardScaler:
    def __init__(self):
        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.energy_mean = np.zeros((1, 1))
        self.energy_std = np.ones((1, 1))
        self.gradient_mean = np.zeros((1, 1, 1, 1))
        self.gradient_std = np.ones((1, 1, 1, 1))

        self._encountered_y_shape = [None, None]
        self._encountered_y_std = [None, None]

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            energy = y[0]
            gradient = y[1]
            out_e = (energy - self.energy_mean) / self.energy_std
            out_g = gradient / self.gradient_std
            y_res = [out_e, out_g]
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        if y is not None:
            energy = y[0]
            gradient = y[1]
            out_e = energy * self.energy_std + self.energy_mean
            out_g = gradient * self.gradient_std
            y_res = [out_e, out_g]
        return x_res, y_res

    def fit(self, x=None, y=None, auto_scale=None):
        if auto_scale is None:
            auto_scale = {'x_mean': True, 'x_std': True, 'energy_std': True, 'energy_mean': True}

        npeps = np.finfo(float).eps
        if auto_scale['x_mean']:
            self.x_mean = np.mean(x)
        if auto_scale['x_std']:
            self.x_std = np.std(x) + npeps
        if auto_scale['energy_mean']:
            y1 = y[0]
            self.energy_mean = np.mean(y1, axis=0, keepdims=True)
        if auto_scale['energy_std']:
            y1 = y[0]
            self.energy_std = np.std(y1, axis=0, keepdims=True) + npeps
        self.gradient_std = np.expand_dims(np.expand_dims(self.energy_std, axis=-1), axis=-1) / self.x_std + npeps
        self.gradient_mean = np.zeros_like(self.gradient_std, dtype=np.float32)  # no mean shift expected

        self._encountered_y_shape = [np.array(y[0].shape), np.array(y[1].shape)]
        self._encountered_y_std = [np.std(y[0], axis=0), np.std(y[1], axis=(0, 2, 3))]

    def fit_transform(self, x=None, y=None, auto_scale=None):
        self.fit(x=x,y=y,auto_scale=auto_scale)
        return self.transform(x=x,y=y)

    def save(self, filepath):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'energy_mean': self.energy_mean.tolist(),
                   'energy_std': self.energy_std.tolist(),
                   'gradient_mean': self.gradient_mean.tolist(),
                   'gradient_std': self.gradient_std.tolist()
                   }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)

        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def get_params(self):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'energy_mean': self.energy_mean.tolist(),
                   'energy_std': self.energy_std.tolist(),
                   'gradient_mean': self.gradient_mean.tolist(),
                   'gradient_std': self.gradient_std.tolist()
                   }
        return outdict

    def set_params(self, indict):
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.energy_mean = np.array(indict['energy_mean'])
        self.energy_std = np.array(indict['energy_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def print_params_info(self):
        print("Info: Total-Data gradient std", self._encountered_y_shape[1], ":", self._encountered_y_std[1])
        print("Info: Total-Data energy std", self._encountered_y_shape[0], ":", self._encountered_y_std[0])
        print("Info: Using energy-std", self.energy_std.shape, ":", self.energy_std[0])
        print("Info: Using energy-mean", self.energy_mean.shape, ":", self.energy_mean[0])
        print("Info: Using gradient-std", self.gradient_std.shape, ":", self.gradient_std[0, :, 0, 0])
        print("Info: Using gradient-mean", self.gradient_mean.shape, ":", self.gradient_mean[0, :, 0, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)


class GradientStandardScaler:
    def __init__(self):
        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.gradient_mean = np.zeros((1, 1, 1, 1))
        self.gradient_std = np.ones((1, 1, 1, 1))

        self._encountered_y_shape = None
        self._encountered_y_std = None

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            y_res = (y - self.gradient_mean) / self.gradient_std
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        x_res = x
        out_gradient = y
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        if y is not None:
            out_gradient = y * self.gradient_std + self.gradient_mean
        return x_res, out_gradient

    def fit(self, x, y, auto_scale=None):
        if auto_scale is None:
            auto_scale = {'x_mean': False, 'x_std': False, 'gradient_std': True, 'gradient_mean': False}

        npeps = np.finfo(float).eps
        if auto_scale['x_mean']:
            self.x_mean = np.mean(x)
        if auto_scale['x_std']:
            self.x_std = np.std(x) + npeps
        if auto_scale['gradient_std']:
            self.gradient_std = np.std(y, axis=(0, 3), keepdims=True) + npeps
            self.gradient_mean = np.zeros_like(self.gradient_std)

        self._encountered_y_std = np.std(y, axis=(0, 3), keepdims=True)
        self._encountered_y_shape = np.array(y.shape)

    def fit_transform(self, x=None, y=None, auto_scale=None):
        self.fit(x=x,y=y,auto_scale=auto_scale)
        return self.transform(x=x,y=y)

    def save(self, filepath):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'gradient_mean': self.gradient_mean.tolist(),
                   'gradient_std': self.gradient_std.tolist()
                   }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)

        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def get_params(self):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'gradient_mean': self.gradient_mean.tolist(),
                   'gradient_std': self.gradient_std.tolist(),
                   }
        return outdict

    def set_params(self, indict):
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.gradient_mean = np.array(indict['gradient_mean'])
        self.gradient_std = np.array(indict['gradient_std'])

    def print_params_info(self):

        print("Info: All-data gradient std", self._encountered_y_shape, ":", self._encountered_y_std[0, :, :, 0])
        print("Info: Using gradient-std", self.gradient_std.shape, ":", self.gradient_std[0, :, :, 0])
        print("Info: Using gradient-mean", self.gradient_mean.shape, ":", self.gradient_mean[0, :, :, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)
