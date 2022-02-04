import json

import numpy as np


class SegmentStandardScaler:
    def __init__(self, segments=None):
        self.feat_mean = np.zeros((1, 1))
        self.feat_std = np.ones((1, 1))

        self.segments = segments
        self._encountered_y_shape = None

    def fit(self, y=None, segments=None):
        if segments is not None:
            self.segments = segments

        if self.segments is None:
            raise ValueError("Please define segments to scale features for shape", self.feat_mean.shape)

        feat_std = []
        feat_mean = []
        splits = np.concatenate([np.array([0]), np.cumsum(self.segments)])
        print(splits)
        for i in range(len(self.segments)):
            sub_array = y[:, splits[i]:splits[i + 1]]
            feat_std.append(np.std(sub_array))
            feat_mean.append(np.mean(sub_array))

        feat_mean = np.repeat(np.array(feat_mean), np.array(self.segments))
        feat_std = np.repeat(np.array(feat_std), np.array(self.segments))
        self.feat_std = np.expand_dims(feat_std, axis=0)
        self.feat_mean = np.expand_dims(feat_mean, axis=0)

        self._encountered_y_shape = np.array(y.shape)
        # print(feat_mean,feat_std)

    def transform(self, y=None):
        y_res = None
        if y is not None:
            y_res = (y - self.feat_mean) / self.feat_std
        return y_res

    def inverse_transform(self, y=None):
        y_res = y
        if y is not None:
            y_res = y * self.feat_std + self.feat_mean
        return y_res

    def fit_transform(self, y=None, segments=None):
        self.fit(y=y,segments=segments)
        return self.transform(y=y)

    def save(self, filepath):
        outdict = {'feat_mean': self.feat_mean.tolist(),
                   'feat_std': self.feat_std.tolist(),
                   }
        with open(filepath, 'w') as f:
            json.dump(outdict, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            indict = json.load(f)

        self.feat_mean = np.array(indict['feat_mean'])
        self.feat_std = np.array(indict['feat_std'])

    def get_params(self):
        outdict = {'feat_mean': self.feat_mean.tolist(),
                   'feat_std': self.feat_std.tolist(),
                   }
        return outdict

    def set_params(self, indict):
        self.feat_mean = np.array(indict['feat_mean'])
        self.feat_std = np.array(indict['feat_std'])

    def print_params_info(self):
        print("Info: Data feature shape", self._encountered_y_shape)
        print("Info: Using feature-scale", self.feat_std.shape, ":", self.feat_std)
        print("Info: Using feature-offset", self.feat_mean.shape, ":", self.feat_mean)
