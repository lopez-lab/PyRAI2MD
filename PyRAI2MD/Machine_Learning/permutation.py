######################################################
#
# PyRAI2MD 2 module for utility tools - permutation
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import os
import numpy as np

def PermuteMap(x,y_dict,permute_map,val_split):
    ## This function permute data following the map P.
    ## x is M x N x 3, M entries, N atoms, x,y,z
    ## y_dict has possible three keys 'energy_gradient', 'nac' and 'soc'
    ## energy is M x n, M batches, n states
    ## gradient is M x n x N x 3, M batches, n states, N atoms, x, y, z
    ## nac is M x m x N x 3, M batches, m state pairs, N atoms, x, y, z
    ## soc is M x l, M batches, l state pairs
    ## permute_map is a file including all permutation

    # early stop the function
    if permute_map == 'No':
        return x, y_dict
    if permute_map != 'No' and os.path.exists(permute_map) == False:
       	return x, y_dict

    # load permutation map
    P = np.loadtxt(permute_map) - 1
    P = P.astype(int)
    if len(P.shape) == 1:
        P = P.reshape((1, -1))

    x_new=np.zeros([0,x.shape[1],x.shape[2]]) # initialize coordinates list
    y_dict_new={}

    per_eg = 0
    per_nac = 0
    per_soc = 0

    # pick energy and gradient, note permutation does not change energy
    if 'energy_gradient' in y_dict.keys(): # check energy gradient
        energy = y_dict['energy_gradient'][0]
        grad = y_dict['energy_gradient'][1]
 
        # initialize energy and gradient list
        y_dict_new['energy_gradient'] = [np.zeros([0,energy.shape[1]]),
                                         np.zeros([0,grad.shape[1],grad.shape[2],grad.shape[3]])] 
        per_eg = 1

    # pick nac
    if 'nac' in y_dict.keys():
       	nac = y_dict['nac']

        # initialize nac list
        y_dict_new['nac'] = np.zeros([0,nac.shape[1],nac.shape[2],nac.shape[3]])                 
        per_nac = 1

    # pick soc, not permutation does not change soc
    if 'soc' in y_dict.keys():
        soc = y_dict['soc']

        # initialize soc list
        y_dict_new['soc'] = np.zeros([0,soc.shape[1]])

    kfold = np.ceil(1/val_split).astype(int)
    portion = int(len(x)*val_split)

    ## determine the range of k-fold
    kfoldrange = []
    for k in range(kfold):
        if k < kfold - 1:
            kfoldrange.append([k * portion, (k + 1) * portion])
        else:
            kfoldrange.append([k * portion, len(x)])

    ## permute data per k-fold
    for k in kfoldrange:
        # separate data in kfold
        a, b = k
        kx = x[a: b]
        new_x = kx
        if per_eg == 1:
            kenergy = energy[a: b]
            kgrad = grad[a: b]
            new_e = kenergy
            new_g = kgrad
        if per_nac == 1:
            knac = nac[a: b]
            new_n = knac
        if per_soc == 1:
            ksoc = soc[a: b]
            new_s = ksoc

        for index in P:
            # permute coord along N atoms
            per_x = kx[:,index,:]
            new_x = np.concatenate((new_x, per_x), axis=0)
            if per_eg == 1:
                # permute grad along N atoms
                per_e = kenergy
                per_g = kgrad[:,:,index,:]              
                new_e = np.concatenate((new_e, per_e), axis=0)
                new_g = np.concatenate((new_g, per_g), axis=0)
            if per_nac == 1:
                # permute nac along N atoms
                per_n = knac[:,:,index,:]
                new_n = np.concatenate((new_n, per_n), axis=0)
            if per_soc == 1:
                per_s = ksoc
                new_s = np.concatenate((new_s, per_s), axis=0)

        # merge the new data
        x_new=np.concatenate((x_new,new_x),axis=0)
        if per_eg == 1:   
            y_dict_new['energy_gradient'][0] = np.concatenate((y_dict_new['energy_gradient'][0], new_e), axis = 0)
            y_dict_new['energy_gradient'][1] = np.concatenate((y_dict_new['energy_gradient'][1], new_g), axis = 0)
        if per_nac == 1:
            y_dict_new['nac'] = np.concatenate((y_dict_new['nac'], new_n), axis = 0)
        if per_soc == 1:
            y_dict_new['soc'] = np.concatenate((y_dict_new['soc'], new_s), axis = 0)

    return x_new, y_dict_new
