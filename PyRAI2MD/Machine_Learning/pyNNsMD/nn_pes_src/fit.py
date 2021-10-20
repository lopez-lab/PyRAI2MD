"""
Main interface to start training_??.py scripts in parallel. This can be solved in many different ways.

Possible are server solutions with slurm and MPI. Here only python subprocess are started to local machine.
The training scripts are supposed to read all necessary information from folder. 
NOTE: Path information of folder and training scripts as well as os info are made fetchable but could fail in certain
circumstances.
"""
import os
import subprocess
import sys

from pyNNsMD.nn_pes_src.selection import get_path_for_fit_script


def fit_model_get_python_cmd_os():
    """
    Return proper commandline command for pyhton depending on os.

    Returns:
        str: Python command either python or pyhton3.

    """
    # python or python3 to run
    if sys.platform[0:3] == 'win':
        return 'python'  # or 'python.exe'
    else:
        return 'python3'


def fit_model_by_modeltype(model_type, dist_method, i, filepath, g, m):
    """
    Run the training script in subprocess.

    Args:
        model_type (str): Name of the model.
        dist_method (tba): Method to call training scripts on cluster.
        i (int): Index of model.
        filepath (str): Filepath to model.
        g (int): GPU index to use.
        m (str): Fitmode.

    Returns:
        None.

    """
    print("Run:", filepath, "Instance:", i, "on GPU:", g, m)
    py_script = get_path_for_fit_script(model_type)
    py_cmd = fit_model_get_python_cmd_os()
    if not os.path.exists(py_script):
        print("Error: Can not find trainingsript, please check path", py_script)
    if dist_method:
        proc = subprocess.Popen([py_cmd, py_script, "-i", str(i), '-f', filepath, "-g", str(g), '-m', str(m)])
        return proc
    if not dist_method:
        proc = subprocess.run([py_cmd, py_script, "-i", str(i), '-f', filepath, "-g", str(g), '-m', str(m)],
                              capture_output=False, shell=False)
        return proc
