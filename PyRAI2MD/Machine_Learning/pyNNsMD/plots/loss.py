import os

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
        train_loss,
        val_loss,
        test_loss=None,
        test_step=1,
        val_step=1,
        save_plot_to_file=False,
        dir_save="",
        filename='fit',
        filetypeout='.png',
        unit_loss='#',
        loss_name="MAE",
        plot_title="MAE vs. epochs",
        label_curves="loss"
):
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    if len(train_loss.shape) == 1:
        train_loss = np.expand_dims(train_loss, axis=0)
    if len(val_loss.shape) == 1:
        val_loss = np.expand_dims(val_loss, axis=0)
    if isinstance(label_curves, str):
        label_curves = [label_curves]

    # Training curve
    fig = plt.figure()
    for i in range(train_loss.shape[0]):
        if i < len(label_curves):
            temp_label = label_curves[i]
        else:
            temp_label = "loss"
        plt.plot(np.arange(1, len(train_loss[i]) + 1), train_loss[i], label='Training ' + temp_label, color='c')
    for i in range(train_loss.shape[0]):
        if i < len(label_curves):
            temp_label = label_curves[i]
        else:
            temp_label = "loss"
        plt.plot(np.array(range(1, len(val_loss[i]) + 1)) * val_step, val_loss[i], label='Valid ' + temp_label,
                 color='b')
    plt.xlabel('Epochs')
    plt.ylabel(loss_name + " [" + unit_loss + "]")
    plt.title(plot_title)
    plt.legend(loc='upper right', fontsize='x-large')

    if save_plot_to_file:
        if not os.path.exists(dir_save):
            print("Error: Output directory does not exist")
            return
        outname = os.path.join(dir_save, filename + "_loss" + filetypeout)
        plt.savefig(outname)

    return fig


def plot_learning_curve(learningall,
                        filename='fit',
                        dir_save="",
                        filetypeout='.png'
                        ):
    # Learing rate
    fig = plt.figure()
    plt.plot(np.arange(1, len(learningall) + 1), learningall, label='Learning rate', color='r')
    plt.xlabel("Epochs")
    plt.ylabel('Learning rate')
    plt.title("Learning rate decrease")

    outname = os.path.join(dir_save, filename + "_lr" + filetypeout)
    plt.savefig(outname)

    return fig
