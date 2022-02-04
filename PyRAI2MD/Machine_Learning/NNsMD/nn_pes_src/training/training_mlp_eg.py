"""
The main training script for energy gradient model. Called with ArgumentParse.
"""
# from sklearn.utils import shuffle
# import time
import matplotlib as mpl
import numpy as np
import tensorflow as tf

mpl.use('Agg')
import os
import json
import pickle
import sys
sys.path.append('%s/PyRAI2MD/Machine_Learning' % os.environ['PYRAI2MD'])

import argparse

parser = argparse.ArgumentParser(description='Train a energy-gradient model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus", default=-1, required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode", default="training", required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())
# args = {"filepath":"E:/Benutzer/Patrick/PostDoc/Projects ML/NeuralNet4/NNfit0/energy_gradient_0",'index' : 0,"gpus":0}


fstdout = open(os.path.join(args['filepath'], "fitlog_" + str(args['index']) + ".txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout

print("Input argpars:", args)

from NNsMD.nn_pes_src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))

from NNsMD.utils.callbacks import EarlyStopping, lr_lin_reduction, lr_exp_reduction, lr_step_reduction
from NNsMD.models.mlp_eg import EnergyGradientModel
# from NNsMD.nn_pes_src.legacy import compute_feature_derivative
from NNsMD.datasets.general import load_hyp
from NNsMD.datasets.general import split_validation_training_index
# from NNsMD.nn_pes_src.scaler import save_std_scaler_dict
from NNsMD.scaler.energy import EnergyGradientStandardScaler
from NNsMD.scaler.general import SegmentStandardScaler
from NNsMD.utils.loss import get_lr_metric, ScaledMeanAbsoluteError, r2_metric, ZeroEmptyLoss
from NNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from NNsMD.plots.pred import plot_scatter_prediction
from NNsMD.plots.error import plot_error_vec_mean, plot_error_vec_max


def train_model_energy_gradient(i=0, outdir=None, mode='training'):
    """
    Train an energy plus gradient model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        outdir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for (energy,gradient).

    """
    i = int(i)
    # Load everything from folder
    try:
        with open(os.path.join(outdir, 'data_y'), 'rb') as f:
            y = pickle.load(f)
        with open(os.path.join(outdir, 'data_x'), 'rb') as f:
            x = pickle.load(f)
    except:
        print("Error: Can not load data for fit", outdir)
        return
    hyperall = None
    try:
        hyperall = load_hyp(os.path.join(outdir, 'hyper' + '_v%i' % i + ".json"))
    except:
        print("Error: Can not load hyper for fit", outdir)

    scaler = EnergyGradientStandardScaler()
    try:
        scaler.load(os.path.join(outdir, 'scaler' + '_v%i' % i + ".json"))
    except:
        print("Error: Can not load scaler info for fit", outdir)

    # Model
    hypermodel = hyperall['model']
    # plots
    unit_label_energy = hyperall['plots']['unit_energy']
    unit_label_grad = hyperall['plots']['unit_gradient']
    # Fit
    hyper = hyperall[mode]
    energies_only = hyper['energy_only']
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split']
    initialize_weights = hyper['initialize_weights']
    learning_rate = hyper['learning_rate']
    loss_weights = hyper['loss_weights']
    auto_scale = hyper['auto_scaling']
    normalize_feat = int(hyper['normalization_mode'])
    # step
    use_step_callback = hyper['step_callback']
    use_linear_callback = hyper['linear_callback']
    use_exp_callback = hyper['exp_callback']
    use_early_callback = hyper['early_callback']

    # Data Check here:
    if (len(x.shape) != 3):
        raise ValueError("Input x-shape must be (batch,atoms,3)")
    else:
        print("Found x-shape of", x.shape)
    if (isinstance(y, list) == False):
        raise ValueError("Input y must be list of [energy,gradient]")
    if (len(y[0].shape) != 2):
        raise ValueError("Input energy-shape must be (batch,states)")
    else:
        print("Found energy-shape of", y[0].shape)
    if (len(y[1].shape) != 4):
        raise ValueError("Input gradient-shape must be (batch,states,atoms,3)")
    else:
        print("Found gradient-shape of", y[1].shape)

    # Fit stats dir
    dir_save = os.path.join(outdir, "fit_stats")
    os.makedirs(dir_save, exist_ok=True)

    # cbks,Learning rate schedule
    cbks = []
    if use_early_callback['use']:
        es_cbk = EarlyStopping(**use_early_callback)
        cbks.append(es_cbk)
    if use_linear_callback['use']:
        lr_sched = lr_lin_reduction(**use_linear_callback)
        lr_cbk = tf.keras.callbacks.LearningRateScheduler(lr_sched)
        cbks.append(lr_cbk)
    if use_exp_callback['use']:
        lr_exp = lr_exp_reduction(**use_exp_callback)
        exp_cbk = tf.keras.callbacks.LearningRateScheduler(lr_exp)
        cbks.append(exp_cbk)
    if use_step_callback['use']:
        lr_step = lr_step_reduction(**use_step_callback)
        step_cbk = tf.keras.callbacks.LearningRateScheduler(lr_step)
        cbks.append(step_cbk)

    # Index train test split
    lval = int(len(x) * val_split)
    allind = np.arange(0, len(x))
    i_train, i_val = split_validation_training_index(allind, lval, val_disjoint, i)
    print("Info: Train-Test split at Train:", len(i_train), "Test", len(i_val), "Total", len(x))

    # Make all Model
    out_model = EnergyGradientModel(**hypermodel)
    out_model.precomputed_features = True
    out_model.output_as_dict = True
    out_model.energy_only = energies_only

    # Look for loading weights
    npeps = np.finfo(float).eps
    if not initialize_weights:
        try:
            out_model.load_weights(os.path.join(outdir, "weights" + '_v%i' % i + '.h5'))
            print("Info: Load old weights at:", os.path.join(outdir, "weights" + '_v%i' % i + '.h5'))
            print("Info: Transferring weights...")
        except:
            print("Error: Can't load old weights...")
    else:
        print("Info: Making new initialized weights.")

    # Scale x,y
    scaler.fit(x, y, auto_scale=auto_scale)
    x_rescale, y_rescale = scaler.transform(x, y)
    y1, y2 = y_rescale

    # Model + Model precompute layer +feat
    feat_x, feat_grad = out_model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)
    # Finding Normalization
    feat_x_mean, feat_x_std = out_model.set_const_normalization_from_features(feat_x,normalization_mode=normalize_feat)

    # Train Test split
    xtrain = [feat_x[i_train], feat_grad[i_train]]
    ytrain = [y1[i_train], y2[i_train]]
    xval = [feat_x[i_val], feat_grad[i_val]]
    yval = [y1[i_val], y2[i_val]]

    # Setting constant feature normalization
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    mae_energy = ScaledMeanAbsoluteError(scaling_shape=scaler.energy_std.shape)
    mae_force = ScaledMeanAbsoluteError(scaling_shape=scaler.gradient_std.shape)
    mae_energy.set_scale(scaler.energy_std)
    mae_force.set_scale(scaler.gradient_std)
    if energies_only:
        train_loss = {'energy': 'mean_squared_error', 'force' : ZeroEmptyLoss()}
    else:
        train_loss = {'energy': 'mean_squared_error', 'force': 'mean_squared_error'}
    out_model.compile(optimizer=optimizer,
                      loss=train_loss, loss_weights=loss_weights,
                      metrics={'energy': [mae_energy, lr_metric, r2_metric],
                               'force': [mae_force, lr_metric, r2_metric]})

    scaler.print_params_info()
    print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
    print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)

    print("")
    print("Start fit.")
    out_model.summary()
    hist = out_model.fit(x=xtrain, y={'energy': ytrain[0], 'force': ytrain[1]}, epochs=epo, batch_size=batch_size,
                         callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, {'energy': yval[0], 'force': yval[1]}), verbose=2)
    print("End fit.")
    print("")
    out_model.energy_only = False

    try:
        outname = os.path.join(dir_save, "history_" + ".json")
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
        print("Warning: Cant save history")

    try:
        out_model.save_weights(os.path.join(outdir, "weights" + '_v%i' % i + '.h5'))
        # print(out_model.get_weights())
    except:
        print("Warning: Cant save weights")

    try:
        print("Info: Saving auto-scaler to file...")
        scaler.save(os.path.join(outdir, "scaler" + '_v%i' % i + '.json'))
    except:
        print("Error: Can not export scaler info. Model prediciton will be wrongly scaled.")

    try:
        # Plot and Save
        yval_plot = [y[0][i_val], y[1][i_val]]
        ytrain_plot = [y[0][i_train], y[1][i_train]]
        # Convert back scaler
        pval = out_model.predict(xval)
        ptrain = out_model.predict(xtrain)
        _, pval = scaler.inverse_transform(y=[pval['energy'], pval['force']])
        _, ptrain = scaler.inverse_transform(y=[ptrain['energy'], ptrain['force']])

        print("Info: Predicted Energy shape:", ptrain[0].shape)
        print("Info: Predicted Gradient shape:", ptrain[1].shape)
        print("Info: Plot fit stats...")

        # Plot
        plot_loss_curves([hist.history['energy_mean_absolute_error'], hist.history['force_mean_absolute_error']],
                         [hist.history['val_energy_mean_absolute_error'],
                          hist.history['val_force_mean_absolute_error']],
                         label_curves=["energy", "force"],
                         val_step=epostep, save_plot_to_file=True, dir_save=dir_save,
                         filename='fit' + str(i), filetypeout='.png', unit_loss=unit_label_energy, loss_name="MAE",
                         plot_title="Energy")

        plot_learning_curve(hist.history['energy_lr'], filename='fit' + str(i), dir_save=dir_save)

        plot_scatter_prediction(pval[0], yval_plot[0], save_plot_to_file=True, dir_save=dir_save,
                                filename='fit' + str(i) + "_energy",
                                filetypeout='.png', unit_actual=unit_label_energy, unit_predicted=unit_label_energy,
                                plot_title="Prediction Energy")

        plot_scatter_prediction(pval[1], yval_plot[1], save_plot_to_file=True, dir_save=dir_save,
                                filename='fit' + str(i) + "_grad",
                                filetypeout='.png', unit_actual=unit_label_grad, unit_predicted=unit_label_grad,
                                plot_title="Prediction Gradient")

        plot_error_vec_mean([pval[1], ptrain[1]], [yval_plot[1], ytrain_plot[1]],
                            label_curves=["Validation gradients", "Training Gradients"], unit_predicted=unit_label_grad,
                            filename='fit' + str(i) + "_grad", dir_save=dir_save, save_plot_to_file=True,
                            filetypeout='.png', x_label='Gradients xyz * #atoms * #states ',
                            plot_title="Gradient mean error")

        plot_error_vec_max([pval[1], ptrain[1]], [yval_plot[1], ytrain_plot[1]],
                           label_curves=["Validation", "Training"],
                           unit_predicted=unit_label_grad, filename='fit' + str(i) + "_grad",
                           dir_save=dir_save, save_plot_to_file=True, filetypeout='.png',
                           x_label='Gradients xyz * #atoms * #states ', plot_title="Gradient max error")

    except:
        print("Error: Could not plot fitting stats")

    error_val = None
    try:
        # Safe fitting Error MAE
        pval = out_model.predict(xval)
        ptrain = out_model.predict(xtrain)
        _, pval = scaler.inverse_transform(y=[pval['energy'], pval['force']])
        _, ptrain = scaler.inverse_transform(y=[ptrain['energy'], ptrain['force']])
        out_model.precomputed_features = False
        out_model.output_as_dict = False
        ptrain2 = out_model.predict(x_rescale[i_train])
        _, ptrain2 = scaler.inverse_transform(y=[ptrain2[0], ptrain2[1]])
        print("Info: Max error precomputed and full gradient computation:")
        print("Energy", np.max(np.abs(ptrain[0] - ptrain2[0])))
        print("Gradient", np.max(np.abs(ptrain[1] - ptrain2[1])))
        error_val = [np.mean(np.abs(pval[0] - y[0][i_val])), np.mean(np.abs(pval[1] - y[1][i_val]))]
        error_train = [np.mean(np.abs(ptrain[0] - y[0][i_train])), np.mean(np.abs(ptrain[1] - y[1][i_train]))]
        np.save(os.path.join(outdir, "fiterr_valid" + '_v%i' % i + ".npy"), error_val)
        np.save(os.path.join(outdir, "fiterr_train" + '_v%i' % i + ".npy"), error_train)
        print("error_val:", error_val)
        print("error_train:", error_train)
    except:
        print("Error: Can not save fiterror")

    # print("Feature norm: ", out_model.get_layer('feat_std').get_weights())
    return error_val


if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    out = train_model_energy_gradient(args['index'], args['filepath'], args['mode'])

fstdout.close()
