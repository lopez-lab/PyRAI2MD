"""
The main training script for NAC model. Called with ArgumentParse.
"""
import numpy as np
import tensorflow as tf
# from sklearn.utils import shuffle
import matplotlib as mpl
# from sklearn.utils import shuffle
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

parser = argparse.ArgumentParser(description='Train a nac model from data, parameters given in a folder')

parser.add_argument("-i", "--index", required=True, help="Index of the NN to train")
parser.add_argument("-f", "--filepath", required=True, help="Filepath to weights, hyperparameter, data etc. ")
parser.add_argument("-g", "--gpus", default=-1, required=True, help="Index of gpu to use")
parser.add_argument("-m", "--mode", default="training", required=True, help="Which mode to use train or retrain")
args = vars(parser.parse_args())
# args = {"filepath":"E:/Benutzer/Patrick/PostDoc/Projects ML/NeuralNet4/NNfit0/nac_0",'index' : 0,"gpus":0}


fstdout = open(os.path.join(args['filepath'], "fitlog_" + str(args['index']) + ".txt"), 'w')
sys.stderr = fstdout
sys.stdout = fstdout

print("Input argpars:", args)

from NNsMD.nn_pes_src.device import set_gpu

set_gpu([int(args['gpus'])])
print("Logic Devices:", tf.config.experimental.list_logical_devices('GPU'))

from NNsMD.utils.callbacks import EarlyStopping, lr_lin_reduction, lr_exp_reduction, lr_step_reduction
from NNsMD.models.mlp_nac2 import NACModel2
from NNsMD.datasets.general import load_hyp
from NNsMD.datasets.general import split_validation_training_index
from NNsMD.scaler.nac import NACStandardScaler
from NNsMD.scaler.general import SegmentStandardScaler
from NNsMD.utils.loss import ScaledMeanAbsoluteError, get_lr_metric, r2_metric, NACphaselessLoss
from NNsMD.plots.loss import plot_loss_curves, plot_learning_curve
from NNsMD.plots.pred import plot_scatter_prediction
from NNsMD.plots.error import plot_error_vec_mean, plot_error_vec_max


def train_model_nac(i=0, outdir=None, mode='training'):
    """
    Train NAC model. Uses precomputed feature and model representation.

    Args:
        i (int, optional): Model index. The default is 0.
        outdir (str, optional): Direcotry for fit output. The default is None.
        mode (str, optional): Fitmode to take from hyperparameters. The default is 'training'.

    Raises:
        ValueError: Wrong input shape.

    Returns:
        error_val (list): Validation error for NAC.

    """
    i = int(i)
    # Load everything from folder
    try:
        with open(os.path.join(outdir, 'data_y'), 'rb') as f:
            y_in = pickle.load(f)
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

    scaler = NACStandardScaler()
    try:
        scaler.load(os.path.join(outdir, 'scaler' + '_v%i' % i + ".json"))
    except:
        print("Error: Can not load scaler info for fit", outdir)

    # Model
    hypermodel = hyperall['model']
    num_outstates = int(hypermodel['states'])
    indim = int(hypermodel['atoms'])
    # plots
    unit_label_nac = hyperall['plots']['unit_nac']
    # Fit
    hyper = hyperall[mode]
    phase_less_loss = hyper['phase_less_loss']
    epo = hyper['epo']
    batch_size = hyper['batch_size']
    epostep = hyper['epostep']
    pre_epo = hyper['pre_epo']
    val_disjoint = hyper['val_disjoint']
    val_split = hyper['val_split']
    initialize_weights = hyper['initialize_weights']
    learning_rate = hyper['learning_rate']
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
    if (len(y_in.shape) != 4):
        raise ValueError("Input nac-shape must be (batch,states,atoms,3)")
    else:
        print("Found nac-shape of", y_in.shape)

    # Set stat dir
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

    # Data selection
    lval = int(len(x) * val_split)
    allind = np.arange(0, len(x))
    i_train, i_val = split_validation_training_index(allind, lval, val_disjoint, i)
    print("Info: Train-Test split at Train:", len(i_train), "Test", len(i_val), "Total", len(x))

    # Make all Models
    out_model = NACModel2(**hypermodel)
    out_model.precomputed_features = True

    npeps = np.finfo(float).eps
    if (initialize_weights == False):
        try:
            out_model.load_weights(os.path.join(outdir, "weights" + '_v%i' % i + '.h5'))
            print("Info: Load old weights at:", os.path.join(outdir, "weights" + '_v%i' % i + '.h5'))
            print("Info: Transferring weights...")
        except:
            print("Error: Can't load old weights...")
    else:
        print("Info: Making new initialized weights..")

    scaler.fit(x, y_in, auto_scale=auto_scale)
    x_rescale, y = scaler.transform(x=x, y=y_in)

    # Calculate features
    feat_x, feat_grad = out_model.precompute_feature_in_chunks(x_rescale, batch_size=batch_size)

    # Finding Normalization
    feat_x_mean, feat_x_std = out_model.set_const_normalization_from_features(feat_x,normalization_mode=normalize_feat)

    xtrain = [feat_x[i_train], feat_grad[i_train]]
    ytrain = y[i_train]
    xval = [feat_x[i_val], feat_grad[i_val]]
    yval = y[i_val]

    # Set Scaling
    scaled_metric = ScaledMeanAbsoluteError(scaling_shape=scaler.nac_std.shape)
    scaled_metric.set_scale( scaler.nac_std)

    scaler.print_params_info()
    print("Info: Using feature-scale", feat_x_std.shape, ":", feat_x_std)
    print("Info: Using feature-offset", feat_x_mean.shape, ":", feat_x_mean)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    lr_metric = get_lr_metric(optimizer)
    out_model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=[scaled_metric, lr_metric, r2_metric])

    # Pre -fit
    if (pre_epo > 0):
        print("Start Pre-fit without phaseless-loss.")
        print("Used loss:", out_model.loss)
        out_model.summary()
        out_model.fit(x=xtrain, y=ytrain, epochs=pre_epo, batch_size=batch_size, verbose=2)
        print("End fit.")
        print("")

    print("Start fit.")
    if phase_less_loss:
        print("Recompiling with phaseless loss.")
        out_model.compile(
            loss=NACphaselessLoss(number_state=num_outstates, shape_nac=(indim, 3), name='phaseless_loss'),
            optimizer=optimizer,
            metrics=[scaled_metric, lr_metric, r2_metric])
        print("Used loss:", out_model.loss)

    out_model.summary()
    hist = out_model.fit(x=xtrain, y=ytrain, epochs=epo, batch_size=batch_size, callbacks=cbks, validation_freq=epostep,
                         validation_data=(xval, yval), verbose=2)
    print("End fit.")
    print("")

    print("")
    print("Start fit.")

    try:
        print("Info: Saving history...")
        outname = os.path.join(dir_save, "history_" + ".json")
        outhist = {a: np.array(b, dtype=np.float64).tolist() for a, b in hist.history.items()}
        with open(outname, 'w') as f:
            json.dump(outhist, f)
    except:
        print("Warning: Cant save history")

    try:
        # Save Weights
        print("Info: Saving weights...")
        out_model.save_weights(os.path.join(outdir, "weights" + '_v%i' % i + '.h5'))
    except:
        print("Error: Cant save weights")

    try:
        print("Info: Saving auto-scaler to file...")
        scaler.save(os.path.join(outdir, "scaler" + '_v%i' % i + '.json'))
    except:
        print("Error: Can not export scaler info. Model prediciton will be wrongly scaled.")

    try:
        # Plot stats
        yval_plot = y_in[i_val]
        ytrain_plot = y_in[i_train]
        # Revert standard but keep unit conversion
        pval = out_model.predict(xval)
        ptrain = out_model.predict(xtrain)
        _, pval = scaler.inverse_transform(y=pval)
        _, ptrain = scaler.inverse_transform(y=ptrain)

        print("Info: Predicted NAC shape:", ptrain.shape)
        print("Info: Plot fit stats...")

        plot_loss_curves(hist.history['mean_absolute_error'],
                         hist.history['val_mean_absolute_error'],
                         label_curves="NAC",
                         val_step=epostep, save_plot_to_file=True, dir_save=dir_save,
                         filename='fit' + str(i) + "_nac", filetypeout='.png', unit_loss=unit_label_nac,
                         loss_name="MAE",
                         plot_title="NAC")

        plot_learning_curve(hist.history['lr'], filename='fit' + str(i), dir_save=dir_save)

        plot_scatter_prediction(pval, yval_plot, save_plot_to_file=True, dir_save=dir_save,
                                filename='fit' + str(i) + "_nac",
                                filetypeout='.png', unit_actual=unit_label_nac, unit_predicted=unit_label_nac,
                                plot_title="Prediction NAC")

        plot_error_vec_mean([pval, ptrain], [yval_plot, ytrain_plot],
                            label_curves=["Validation NAC", "Training NAC"], unit_predicted=unit_label_nac,
                            filename='fit' + str(i) + "_nac", dir_save=dir_save, save_plot_to_file=True,
                            filetypeout='.png', x_label='NACs xyz * #atoms * #states ',
                            plot_title="NAC mean error")

        plot_error_vec_max([pval, ptrain], [yval_plot, ytrain_plot],
                           label_curves=["Validation", "Training"],
                           unit_predicted=unit_label_nac, filename='fit' + str(i) + "_nc",
                           dir_save=dir_save, save_plot_to_file=True, filetypeout='.png',
                           x_label='NACs xyz * #atoms * #states ', plot_title="NAC max error")
    except:
        print("Warning: Could not plot fitting stats")

    # error out
    error_val = None

    try:
        print("Info: saving fitting error...")
        # Safe fitting Error MAE
        pval = out_model.predict(xval)
        ptrain = out_model.predict(xtrain)
        _, pval = scaler.inverse_transform(y=pval)
        _, ptrain = scaler.inverse_transform(y=ptrain)
        out_model.precomputed_features = False
        ptrain2 = out_model.predict(x_rescale[i_train])
        ptrain2 = ptrain2 * scaler.nac_std + scaler.nac_mean
        print("Info: MAE between precomputed and full keras model:")
        print("NAC", np.mean(np.abs(ptrain - ptrain2)))
        error_val = np.mean(np.abs(pval - y_in[i_val]))
        error_train = np.mean(np.abs(ptrain - y_in[i_train]))
        print("error_val:", error_val)
        print("error_train:", error_train)
        np.save(os.path.join(outdir, "fiterr_valid" + '_v%i' % i + ".npy"), error_val)
        np.save(os.path.join(outdir, "fiterr_train" + '_v%i' % i + ".npy"), error_train)
    except:
        print("Error: Can not save fiterror")

    return error_val


if __name__ == "__main__":
    print("Training Model: ", args['filepath'])
    print("Network instance: ", args['index'])
    train_model_nac(args['index'], args['filepath'], args['mode'])

fstdout.close()
