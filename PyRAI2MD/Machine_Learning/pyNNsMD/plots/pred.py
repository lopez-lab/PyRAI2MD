import os

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter_prediction(
        y_pred,
        y_val,
        save_plot_to_file=False,
        dir_save="",
        filename='fit',
        filetypeout='.png',
        unit_actual='#',
        unit_predicted="#",
        plot_title="Prediction"
):
    fig = plt.figure()

    preds = y_pred.flatten()
    engval = y_val.flatten()
    engval_min = np.amin(engval)
    engval_max = np.amax(engval)
    plt.plot(np.arange(engval_min, engval_max, np.abs(engval_min - engval_max) / 100),
             np.arange(engval_min, engval_max, np.abs(engval_min - engval_max) / 100), color='red')
    plt.scatter(preds, engval, alpha=0.3)
    plt.xlabel('Predicted ' + " [" + unit_predicted + "]")
    plt.ylabel('Actual ' + " [" + unit_actual + "]")
    plt.title(plot_title)
    plt.text(engval_min, engval_max,
             "MAE: {0:0.3f} ".format(np.mean(np.abs(preds - engval))) + "[" + unit_predicted + "]")

    if save_plot_to_file:
        outname = os.path.join(dir_save, filename + "_predict" + filetypeout)
        plt.savefig(outname)

    return fig
