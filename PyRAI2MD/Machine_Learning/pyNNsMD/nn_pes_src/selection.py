"""
Model selection
"""
import os

import numpy as np

from pyNNsMD.models.mlp_e import EnergyModel
from pyNNsMD.models.mlp_eg import EnergyGradientModel
from pyNNsMD.models.mlp_nac import NACModel
from pyNNsMD.models.mlp_nac2 import NACModel2
from pyNNsMD.models.mlp_g2 import GradientModel2

from pyNNsMD.nn_pes_src.hypers.hyper_mlp_e import DEFAULT_HYPER_PARAM_ENERGY
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_eg import DEFAULT_HYPER_PARAM_ENERGY_GRADS
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_g2 import DEFAULT_HYPER_PARAM_GRADS2
from pyNNsMD.nn_pes_src.hypers.hyper_mlp_nac import DEFAULT_HYPER_PARAM_NAC

from pyNNsMD.scaler.energy import EnergyGradientStandardScaler, EnergyStandardScaler, GradientStandardScaler
from pyNNsMD.scaler.nac import NACStandardScaler


def get_default_hyperparameters_by_modeltype(model_type):
    """
    Select the default parameters for each model

    Args:
        model_type (str): Model identifier.

    Returns:
        dict: Default hyper parameters for model.

    """
    model_dict = {'mlp_eg': DEFAULT_HYPER_PARAM_ENERGY_GRADS,
                  'mlp_e': DEFAULT_HYPER_PARAM_ENERGY,
                  'mlp_g2': DEFAULT_HYPER_PARAM_GRADS2,
                  'mlp_nac': DEFAULT_HYPER_PARAM_NAC,
                  'mlp_nac2': DEFAULT_HYPER_PARAM_NAC}
    return model_dict[model_type]


def get_path_for_fit_script(model_type):
    """
    Interface to find the path of training scripts.

    For now they are expected to be in the same folder-system as calling .py script.

    Args:
        model_type (str): Name of the model.

    Returns:
        filepath (str): Filepath pointing to training scripts.

    """
    # Ways of finding path either os.getcwd() or __file__ or just set static path with install...
    # locdiR = os.getcwd()
    filepath = os.path.abspath(os.path.dirname(__file__))
    # STATIC_PATH_FIT_SCRIPT = ""
    fit_script = {"mlp_eg": "training_mlp_eg.py",
                  "mlp_nac": "training_mlp_nac.py",
                  "mlp_nac2": "training_mlp_nac2.py",
                  "mlp_e": "training_mlp_e.py",
                  'mlp_g2' : "training_mlp_g2.py"
                  }
    outpath = os.path.join(filepath, "training", fit_script[model_type])
    return outpath


def get_default_scaler(model_type):
    """
    Get default values for scaler in and output for each model.

    Args:
        model_type (str): Model identifier.

    Returns:
        Dict: Scaling dictionary.

    """
    if model_type == 'mlp_e':
        return EnergyStandardScaler()
    elif model_type == 'mlp_eg':
        return EnergyGradientStandardScaler()
    elif model_type == 'mlp_nac' or model_type == 'mlp_nac2':
        return NACStandardScaler()
    elif model_type == 'mlp_g2':
        return GradientStandardScaler()
    else:
        print("Error: Unknown model type", model_type)
        raise TypeError(f"Error: Unknown model type for default scaler {model_type}")


def get_model_by_type(model_type, hyper):
    """
    Find the implemented model by its string identifier.

    Args:
        model_type (str): Model type.
        hyper (dict): Dict with hyper parameters.

    Returns:
        tf.keras.model: Defult initialized tf.keras.model.

    """
    if model_type == 'mlp_nac':
        return NACModel(**hyper)
    elif model_type == 'mlp_nac2':
        return NACModel2(**hyper)
    elif model_type == 'mlp_eg':
        return EnergyGradientModel(**hyper)
    elif model_type == 'mlp_e':
        return EnergyModel(**hyper)
    elif model_type == 'mlp_g2':
        return GradientModel2(**hyper)
    else:
        print("Error: Unknown model type", model_type)
        raise TypeError(f"Error: Unknown model type forn{model_type}")


def predict_uncertainty(model_type, out, mult_nn):
    if isinstance(out[0], list):
        out_mean = []
        out_std = []
        for i in range(len(out[0])):
            out_mean.append(np.mean(np.array([x[i] for x in out]), axis=0))
            if mult_nn > 1:
                out_std.append(np.std(np.array([x[i] for x in out]), axis=0, ddof=1))
            else:
                out_std.append(np.zeros_like(out_mean[-1]))

        return out_mean, out_std
    else:
        out_mean = np.mean(np.array(out), axis=0)
        if mult_nn > 1:
            out_std = np.std(np.array(out), axis=0, ddof=1)
        else:
            out_std = np.zeros_like(out_mean)
        return out_mean, out_std


def unpack_convert_y_to_numpy(model_type, temp):
    if isinstance(temp, list):
        return [x.numpy() for x in temp]
    else:
        return temp.numpy()
