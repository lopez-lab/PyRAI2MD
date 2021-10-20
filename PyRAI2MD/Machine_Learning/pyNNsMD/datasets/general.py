"""
Shared and general data handling functionality.
"""

import json
import os
import pickle

import numpy as np
from sklearn.utils import shuffle


def index_make_random_shuffle(x):
    """
    Shuffle indexarray.

    Args:
        x (np.array): Index to shuffle.

    Returns:
        np.array: Shuffled index.

    """
    return shuffle(x)


def make_random_shuffle(datalist, shuffle_ind=None):
    """
    Shuffle a list od data.

    Args:
        datalist (list): List of numpy arrays of same length (axis=0).
        shuffle_ind (np.array): Array of shuffled index

    Returns:
        outlist (list): List of the shuffled data.

    """
    datalen = len(datalist[0])  # this should be x data
    for x in datalist:
        if len(x) != datalen:
            print("Error: Data has inconsisten length")

    if shuffle_ind is None:
        allind = shuffle(np.arange(datalen))
    else:
        allind = shuffle_ind
        if len(allind) != datalen:
            print("Warning: Datalength and shuffle index does not match")

    outlist = []
    for x in datalist:
        outlist.append(x[allind])
    return allind, outlist


def save_data_to_folder(x, y, target_model, mod_dir, random_shuffle):
    """
    Save all training data for model mlp_eg to folder.

    Args:
        x (np.array): Coordinates as x-data.
        y (list): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        target_model (str): Name of the Model to save data for.
        mod_dir (str): Path of model directory.
        random_shuffle (bool, optional): Whether to shuffle data before save. The default is False.

    Returns:
        None.

    """
    # Save data:
    if not random_shuffle:
        with open(os.path.join(mod_dir, 'data_x'), 'wb') as f:
            pickle.dump(x, f)
        with open(os.path.join(mod_dir, 'data_y'), 'wb') as f:
            pickle.dump(y, f)
    else:
        if isinstance(y, list):
            shuffle_list = [x] + y
        else:
            shuffle_list = [x] + [y]
        # Make random shuffle
        ind_shuffle, datalist = make_random_shuffle(shuffle_list)
        x_out = datalist[0]
        if len(datalist) > 2:
            y_out = datalist[1:]
        else:
            y_out = datalist[1]
        np.save(os.path.join(mod_dir, 'shuffle_index.npy'), ind_shuffle)
        with open(os.path.join(mod_dir, 'data_x'), 'wb') as f:
            pickle.dump(x_out, f)
        with open(os.path.join(mod_dir, 'data_y'), 'wb') as f:
            pickle.dump(y_out, f)


def split_validation_training_index(allind, splitsize, do_offset, offset_steps):
    """
    Make a train-validation split for indexarray. Validation set is taken from beginning with possible offset.
 
    Args:
        allind (np.array): Indexlist for full dataset of same length.
        splitsize (int): Total number of validation samples to take.
        do_offset (bool): Whether to take validation set not from beginnig but with offset.
        offset_steps (int): Number of validation sizes offseted from the beginning to start to take validation set.

    Returns:
        i_train (np.array): Training indices
        i_val (np.array): Validation indices.

    """
    i = offset_steps
    lval = splitsize
    if not do_offset:
        i_val = allind[:lval]
        i_train = allind[lval:]
    else:
        i_val = allind[i * lval:(i + 1) * lval]
        i_train = np.concatenate([allind[0:i * lval], allind[(i + 1) * lval:]], axis=0)
    if len(i_val) <= 0:
        print("Warning: #Validation data is 0, take 1 training sample instead")
        i_val = i_train[:1]

    return i_train, i_val


def merge_np_arrays_in_chunks(data1, data2, split_size):
    """
    Merge data in chunks of split-size. Goal is to keep validation k-splits for fit.
    
    Idea: [a+a+a] + [b+b+b] = [(a+b)+(a+b)+(a+b)] and NOT [a+a+a+b+b+b].

    Args:
        data1 (np.array): Data to merge.
        data2 (np.array): Data to merge.
        split_size (float): Relative size of junks 0 < split_size < 1.

    Returns:
        np.array: Merged data.

    """
    pacs1 = int(len(data1) * split_size)
    pacs2 = int(len(data2) * split_size)

    data1frac = [data1[i * pacs1:(i + 1) * pacs1] for i in range(int(np.ceil(1 / split_size)))]
    data2frac = [data2[i * pacs2:(i + 1) * pacs2] for i in range(int(np.ceil(1 / split_size)))]

    for i in range(len(data1frac)):
        data1frac[i] = np.concatenate([data1frac[i], data2frac[i]], axis=0)

    return np.concatenate(data1frac, axis=0)


def model_make_random_shuffle(x, y, shuffle_ind):
    """
    Shuffle data according to model.

    Args:
        x (np.array): Coordinates as x-data.
        y (list): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        shuffle_ind (np.array): Index order of datapoints in dataset to shuffle after.

    Returns:
        None.

    """
    if isinstance(y, list):
        _, temp = make_random_shuffle([x] + y, shuffle_ind)
        return temp[0], temp[1:]
    else:
        return make_random_shuffle([x, y], shuffle_ind)[1]


def model_merge_data_in_chunks(mx1, my1, mx2, my2, val_split=0.1):
    """
    Merge Data in chunks.

    Args:
        mx1 (list,np.array): Coordinates as x-data.
        my1 (list,np.array): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        mx2 (list,np.array): Coordinates as x-data.
        my2 (list,np.array): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        val_split (float, optional): Validation split. Defaults to 0.1.

    Raises:
        TypeError: Unkown model type.

    Returns:
        x: Merged x data. Depending on model.
        y: Merged y data. Depending on model.

    """
    if isinstance(my1, list) and isinstance(my2, list):
        x_merge = merge_np_arrays_in_chunks(mx1, mx2, val_split)
        y_merge = [merge_np_arrays_in_chunks(my1[i], my2[i], val_split) for i in range(len(my1))]
        return x_merge, y_merge
    else:
        x_merge = merge_np_arrays_in_chunks(mx1, mx2, val_split)
        y_merge = merge_np_arrays_in_chunks(my1, my2, val_split)
        return x_merge, y_merge


def model_save_data_to_folder(x, y,
                              target_model,
                              mod_dir,
                              random_shuffle=False):
    """
    Save Data to model folder. Always dumps data_x and data_y as pickle.

    Args:
        x (np.array): Coordinates as x-data.
        y (list): A possible list of np.arrays for y-values. Energy, Gradients, NAC etc.
        target_model (str): Name of the Model to save data for.
        mod_dir (str): Path of model directory.
        random_shuffle (bool, optional): Whether to shuffle data before save. The default is False.

    Returns:
        None.
    """
    return save_data_to_folder(x, y, target_model, mod_dir, random_shuffle)


def save_hyp(hyperparameter, filepath):
    """
    Save hyper-parameters as json dict.

    Args:
        hyperparameter: dict of hyperparameter
        filepath: filepath to save data

    Returns:
        None
    """
    with open(filepath, 'w') as f:
        json.dump(hyperparameter, f)


def load_hyp(filepath):
    """
    Load hyper-parameters from filepath

    Args:
        filepath: filepath of .json file loaded into dict.

    Returns:
        dict: hyper-parameters
    """
    with open(filepath, 'r') as f:
        return json.load(f)
