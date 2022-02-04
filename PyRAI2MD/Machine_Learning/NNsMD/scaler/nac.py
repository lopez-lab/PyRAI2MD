import json

import numpy as np


class NACStandardScaler:
    def __init__(self):
        self.x_mean = np.zeros((1, 1, 1))
        self.x_std = np.ones((1, 1, 1))
        self.nac_mean = np.zeros((1, 1, 1, 1))
        self.nac_std = np.ones((1, 1, 1, 1))

        self._encountered_y_shape = None
        self._encountered_y_std = None

    def transform(self, x=None, y=None):
        x_res = x
        y_res = y
        if x is not None:
            x_res = (x - self.x_mean) / self.x_std
        if y is not None:
            y_res = (y - self.nac_mean) / self.nac_std
        return x_res, y_res

    def inverse_transform(self, x=None, y=None):
        x_res = x
        out_nac = y
        if x is not None:
            x_res = x * self.x_std + self.x_mean
        if y is not None:
            out_nac = y * self.nac_std + self.nac_mean
        return x_res, out_nac

    def fit(self, x, y, auto_scale=None):
        if auto_scale is None:
            auto_scale = {'x_mean': False, 'x_std': False, 'nac_std': True, 'nac_mean': False}

        npeps = np.finfo(float).eps
        if auto_scale['x_mean']:
            self.x_mean = np.mean(x)
        if auto_scale['x_std']:
            self.x_std = np.std(x) + npeps
        if auto_scale['nac_std']:
            self.nac_std = np.std(y, axis=(0, 3), keepdims=True) + npeps
            self.nac_mean = np.zeros_like(self.nac_std)

        self._encountered_y_std = np.std(y, axis=(0, 3), keepdims=True)
        self._encountered_y_shape = np.array(y.shape)

    def fit_transform(self, x=None, y=None, auto_scale=None):
        self.fit(x=x,y=y,auto_scale=auto_scale)
        return self.transform(x=x,y=y)

    def save(self, filepath):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'nac_mean': self.nac_mean.tolist(),
                   'nac_std': self.nac_std.tolist()
                   }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)

        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.nac_mean = np.array(indict['nac_mean'])
        self.nac_std = np.array(indict['nac_std'])

    def get_params(self):
        outdict = {'x_mean': self.x_mean.tolist(),
                   'x_std': self.x_std.tolist(),
                   'nac_mean': self.nac_mean.tolist(),
                   'nac_std': self.nac_std.tolist(),
                   }
        return outdict

    def set_params(self, indict):
        self.x_mean = np.array(indict['x_mean'])
        self.x_std = np.array(indict['x_std'])
        self.nac_mean = np.array(indict['nac_mean'])
        self.nac_std = np.array(indict['nac_std'])

    def print_params_info(self):

        print("Info: All-data NAC std", self._encountered_y_shape, ":", self._encountered_y_std[0, :, :, 0])
        print("Info: Using nac-std", self.nac_std.shape, ":", self.nac_std[0, :, :, 0])
        print("Info: Using nac-mean", self.nac_mean.shape, ":", self.nac_mean[0, :, :, 0])
        print("Info: Using x-scale", self.x_std.shape, ":", self.x_std)
        print("Info: Using x-offset", self.x_mean.shape, ":", self.x_mean)
