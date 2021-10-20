"""
Callbacks for learning rate schedules.
"""

import logging
import time

import numpy as np
import tensorflow as tf


def lr_step_reduction(learning_rate_step=[1e-3, 1e-4, 1e-5], epoch_step_reduction=[500, 1000, 5000],use=None):
    """
    Make learning rate schedule function for step reduction.

    Args:
        learnrate_steps (list, optional): List of learning rates for each step. The default is [1e-3,1e-4,1e-5].
        learnrate_epochs (list, optional): The length of each step to keep learning rate.
                                            The default is [500,1000,5000].

    Returns:
        func: Function that can be used with LearningRateScheduler.
        
    Example:
        lr_schedule_steps = tf.keras.callbacks.LearningRateScheduler(lr_step_reduction)

    """
    learning_rate_abs = np.cumsum(np.array(epoch_step_reduction))

    def lr_out_step(epoch):
        # epo = int(learning_rate_abs[-1])
        learning_rate = float(learning_rate_step[-1])
        le = np.array(learning_rate_abs)
        lr = np.array(learning_rate_step)
        out = np.select(epoch <= le, lr, default=learning_rate)
        return float(out)

    return lr_out_step


def lr_lin_reduction(learning_rate_start=1e-3, learning_rate_stop=1e-5, epo=10000, epomin=1000,use=None):
    """
    Make learning rate schedule function for linear reduction.

    Args:
        learning_rate_start (float, optional): Learning rate to start with. The default is 1e-3.
        learning_rate_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epo (int, optional): Total number of epochs to reduce learning rate towards. The default is 10000.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant.
                                The default is 1000.

    Returns:
        func: Function to use with LearningRateScheduler.
    
    Example:
        lr_schedule_lin = tf.keras.callbacks.LearningRateScheduler(lr_lin_reduction)

    """

    def lr_out_lin(epoch):
        if epoch < epomin:
            out = learning_rate_start
        else:
            out = float(
                learning_rate_start - (learning_rate_start - learning_rate_stop) / (epo - epomin) * (epoch - epomin))
        return out

    return lr_out_lin


def lr_exp_reduction(learning_rate_start, epomin, epostep, factor_lr,use=None):
    """
    Make learning rate schedule function for exponential reduction.

    Args:
        learning_rate_start (float): Learning rate to start with.
        epomin (float): Minimum number of epochs to keep learning rate constant.
        epostep (float): The epochs to divide factor by.
        facred (float): Reduce learning rate by factor.

    Returns:
        func: Function to use with LearningRateScheduler.
    
    Example:
        lr_schedule_exp = tf.keras.callbacks.LearningRateScheduler(lr_exp_reduction)

    """

    def lr_out_exp(epo):
        if epo < epomin:
            out = learning_rate_start
        else:
            out = learning_rate_start * np.power(factor_lr, (epo - epomin) / epostep)
        return float(out)

    return lr_out_exp



class EarlyStopping(tf.keras.callbacks.Callback):
    """
    This Callback does basic monitoring of the learning process.
    
    And provides functionality such as learning rate decay and early stopping with custom logic as opposed to the
    callbacks provided by Keras by default which are generic.
    By André Eberhard
    https://github.com/patchmeifyoucan
    """

    def __init__(self,
                 max_time=np.Infinity,
                 epochs=np.Infinity,
                 learning_rate_start=1e-3,
                 epostep=1,
                 loss_monitor='val_loss',
                 delta_loss=0.00001,
                 patience=100,
                 epomin=0,
                 factor_lr=0.5,
                 learning_rate_stop=0.000001,
                 store_weights=False,
                 restore_weights_on_lr_decay=False,
                 use = None
                 ):
        """
        Make Callback for early stopping.
        
        Args:     
        minutes (int): Duration in minutes of training, stops training even if number of epochs is not reached yet.
        epochs (int): Number of epochs to train. stops training even if number of minutes is not reached yet.
        learning_rate (float): The learning rate for the optimizer.
        epostep (int): Step to check for monitor loss.
        monitor (str): The loss quantity to monitor for early stopping operations.
        min_delta (float): Minimum improvement to reach after 'patience' epochs of training.
        patience (int): Number of epochs to wait before decreasing learning rate by a factor of 'factor'.
        min_epoch (int): Minimum Number of epochs to run before decreasing learning rate
        factor (float): new_lr = old_lr * factor
        min_lr (float): Learning rate is not decreased any further after "min_lr" is reached.
        store_weights (bool): If True, stores parameters of best run so far when learning rate is decreased.
        restore_weights_on_lr_decay (bool): If True, restores parameters of best run so far when learning rate is
                                            decreased.
        
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.minutes = max_time
        self.epochs = epochs
        self.epostep = epostep
        self.minEpoch = epomin

        self.start = None
        self.stopped = False
        self.batch_size = None
        self.batch_size_initial = None
        self.learning_rate = learning_rate_start

        self.monitor = loss_monitor
        self.min_delta = delta_loss
        self.factor = factor_lr
        self.patience = patience
        self.min_lr = learning_rate_stop
        self.restore_weights_on_lr_decay = restore_weights_on_lr_decay
        self.store_weights = store_weights

        self.best_weights = None
        self.current_epoch = 0
        self.current_minutes = 0
        self.epochs_without_improvement = 0
        self.best_loss = np.Infinity

    def _reset_weights(self):
        if self.best_weights is None:
            return

        self.logger.info("resetting model weights")
        self.model.set_weights(self.best_weights)

    def _decrease_learning_rate(self):
        old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        new_lr = old_lr * self.factor
        if new_lr < self.min_lr:
            self.logger.info(
                f"Reached learning rate {old_lr:.8f} below acceptable {self.min_lr:.8f} without improvement")
            self.model.stop_training = True
            self.stopped = True
            new_lr = self.min_lr

        self.logger.info(f"setting learning rate from {old_lr:.8f} to {new_lr:.8f}")
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.learning_rate = new_lr

        if self.restore_weights_on_lr_decay is True:
            self._reset_weights()

    def _check_time(self):
        self.current_minutes = np.round((time.time() - self.start) / 60).astype(np.int)
        self.logger.info(f"network trained for {self.current_minutes}/{self.minutes} minutes.")
        if self.current_minutes < self.minutes:
            return

        self.logger.info(f"network trained for {self.current_minutes} minutes. stopping.")
        self.model.stop_training = True
        self.stopped = True

    def _check_loss(self, logs):
        current_loss = logs[self.monitor]

        if current_loss < self.best_loss and self.best_loss - current_loss > self.min_delta:
            diff = self.best_loss - current_loss
            self.logger.info(f"{self.monitor} improved by {diff:.6f} from {self.best_loss:.6f} to {current_loss:.6f}.")
            self.best_loss = current_loss
            if self.store_weights:
                self.best_weights = self.model.get_weights()
            self.epochs_without_improvement = 0
            return

        self.epochs_without_improvement += self.epostep
        if self.epochs_without_improvement < self.patience:
            self.logger.info(f"loss did not improve for {self.epochs_without_improvement} epochs.")
            return

        self.logger.info(f"loss did not improve for max epochs {self.patience}.")
        self.epochs_without_improvement = 0
        self._decrease_learning_rate()

    def on_train_begin(self, logs=None):

        if self.start is None:
            self.start = time.time()
            self.model.summary()
            return

        self._check_time()

    def on_train_end(self, logs=None):
        if self.stopped:
            self._reset_weights()

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        if self.current_epoch % self.epostep == 0:
            self._check_time()
            if self.current_epoch > self.minEpoch:
                self._check_loss(logs)
            loss_diff = logs['val_loss'] - logs['loss']
            self.logger.info(f'current loss_diff: {loss_diff:.6f}')

        if self.current_epoch > self.epochs:
            self.model.stop_training = True
            self.stopped = True
