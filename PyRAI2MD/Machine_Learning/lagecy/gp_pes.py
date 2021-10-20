### Gaussian Process for Photochemical Reaction Prediction
### Original code by Andre Eberhard
### Transplant here Jingbai Li Jun 2 2020

import time,datetime,json
import numpy as np
import logging
import os
import pickle
import sys
import time
from multiprocessing import Pool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


#def get_logger(file):
#    logger = logging.getLogger(str(os.path.basename(file)).rjust(15))
#    log_formatter = logging.Formatter(f'%(asctime)s %(levelname)s %(name)s: %(message)s')
#
#    console_handler = logging.StreamHandler(sys.stdout)
#    console_handler.setFormatter(log_formatter)
#    logger.addHandler(console_handler)
#    logger.setLevel(logging.DEBUG)
#    return logger


class GaussianProcessPes:
    def __init__(self):
        #self._logger = get_logger(__file__)

        # noinspection PyTypeChecker
        self._models: dict = None

    def _create_models(self, y_dict):
        #self._logger.debug("creating models")
        kernel = RBF() * ConstantKernel() + WhiteKernel()
        models = {key: GaussianProcessRegressor(kernel=kernel, normalize_y=value.shape[1] == 1) for key, value in y_dict.items()}
        return models

    def _fit_model(self, model_name, x, y):
        #self._logger.info(f"fitting model {model_name}")
        model: GaussianProcessRegressor = self._models[model_name]
        model.fit(x, y)
        model.kernel = model.kernel_
        return model_name, model

    def _fit_models(self, x, y_dict,n_processes):
        models_available = sorted(list(self._models.keys()))
        models_to_train = sorted(list(y_dict.keys()))

        if models_available != models_to_train:
            raise TypeError(f"Cannot train on data: {models_to_train} does not match models {models_available}!")

        #self._logger.debug(f"fitting models {models_to_train}")
        params = [(model_name, x, y) for model_name, y in y_dict.items()]
        with Pool(n_processes) as p:
            trained_models = p.starmap(self._fit_model, params)

        self._models = {model_name: model_object for model_name, model_object in trained_models}

        #self._logger.debug(f"successfully fitted models {models_to_train}")

    def fit(self, x, y,n_processes=1) -> "GaussianProcessPes":
        if self._models is None:
            self._models = self._create_models(y)

        self._fit_models(x,y,n_processes)

        return self

    def save(self,filename) -> "GaussianProcessPes":
        if self._models is None:
            raise TypeError("Cannot save model before init.")

        #self._logger.debug(f"saving model to {filename}")
        with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(self._models, file, protocol=pickle.HIGHEST_PROTOCOL)

        return self._models

    def load(self, filename) -> "GaussianProcessPes":
        #self._logger.debug(f"loading fitted model")
        with open(f"{filename}", "rb") as file:
            self._models = pickle.load(file)

        return self

    def retrieve(self,modelfile) -> "GaussianProcessPes": 
        self._models=modelfile

        return self

    def _predictions(self,model_name,model,x):
        results=model.predict(x,return_std=True)
        return model_name,results

    def predict(self, x,n_processes=1) -> dict:

        params = [(model_name,model,x) for model_name, model in self._models.items()]
        with Pool(n_processes) as p:
            predictions = p.starmap(self._predictions, params)

        result = {name:results for name, results in predictions}

        y_pred = {key: result[key][0] for key in result.keys()}
        y_std = {key: result[key][1] for key in result.keys()}

        return y_pred, y_std

