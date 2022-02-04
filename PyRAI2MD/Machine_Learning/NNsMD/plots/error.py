import os

import matplotlib.pyplot as plt
import numpy as np


def find_max_relative_error(preds, yval):
    """
    Find maximum error and its relative value if possible.

    Args:
        preds (np.array): Prediction array.
        yval (np.array): Validation array.

    Returns:
        pred_err (np.array): Flatten maximum error along axis=0
        prelm (np.array): Flatten Relative maximum error along axis=0

    """
    pred = np.reshape(preds, (preds.shape[0], -1))
    flat_yval = np.reshape(yval, (yval.shape[0], -1))
    maxerr_ind = np.expand_dims(np.argmax(np.abs(pred - flat_yval), axis=0), axis=0)
    pred_err = np.abs(np.take_along_axis(pred, maxerr_ind, axis=0) -
                      np.take_along_axis(flat_yval, maxerr_ind, axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        prelm = pred_err / np.abs(np.take_along_axis(flat_yval, maxerr_ind, axis=0))
    pred_err = pred_err.flatten()
    prelm = prelm.flatten()
    return pred_err, prelm


def plot_error_vec_mean(
        y_pred,
        y_true,
        label_curves="Vector",
        unit_predicted="#",
        filename='fit',
        dir_save="",
        save_plot_to_file=False,
        filetypeout='.png',
        x_label="Vector components",
        plot_title="Component mean error"
):
    # Forces mean
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if not isinstance(y_true, list):
        y_true = [y_true]
    if isinstance(label_curves, str):
        label_curves = [label_curves]

    fig = plt.figure()
    for i in range(len(y_pred)):
        preds = np.mean(np.abs(y_pred[i] - y_true[i]), axis=0).flatten()
        if i < len(label_curves):
            temp_label = label_curves[i]
        else:
            temp_label = "Vector"
        plt.plot(np.arange(len(preds)), preds, label=temp_label)

    plt.ylabel('Mean absolute error ' + "[" + unit_predicted + "]")
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.title(plot_title)

    if save_plot_to_file:
        outname = os.path.join(dir_save, filename + "_mean" + filetypeout)
        plt.savefig(outname)

    return fig


def plot_error_vec_max(y_pred,
                       y_true,
                       label_curves="Vector",
                       unit_predicted="#",
                       filename='fit',
                       dir_save="",
                       save_plot_to_file=False,
                       filetypeout='.png',
                       x_label="Vector components",
                       plot_title="Component max error"):
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if not isinstance(y_true, list):
        y_true = [y_true]
    if isinstance(label_curves, str):
        label_curves = [label_curves]

    err_max = []
    err_rel = []
    for i in range(len(y_pred)):
        temp_err, temp_rel = find_max_relative_error(y_pred[i], y_true[i])
        err_max.append(temp_err)
        err_rel.append(temp_rel)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for i in range(len(err_max)):
        if i < len(label_curves):
            temp_label = label_curves[i]
        else:
            temp_label = "Vector"
        ax1.plot(np.arange(len(err_max[i])), err_max[i], label="Max " + temp_label)
    plt.ylabel('Max absolute error ' + "[" + unit_predicted + "]")
    plt.legend(loc='upper left')
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    for i in range(len(err_rel)):
        if i < len(label_curves):
            temp_label = label_curves[i]
        else:
            temp_label = "Vector"
        ax2.plot(np.arange(len(err_rel[i])), err_rel[i], label='Rel. ' + temp_label)

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel("Relative max error")
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.title(plot_title)

    if save_plot_to_file:
        outname = os.path.join(dir_save, filename + "_max" + filetypeout)
        plt.savefig(outname)

    return fig1
