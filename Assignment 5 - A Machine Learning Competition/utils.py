#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mai 03, 10:37:26
@last modified : 2021 mei 07, 22:03:30
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from colorama import Fore, Back, Style, init


init(autoreset=False)


class Printer:
    SECTION = -1
    SUBSECTION = -1

    @classmethod
    def print(cls, *args, **kwargs):
        print("> ", *args, **kwargs)

    @classmethod
    def section(cls, *args, **kwargs):
        Printer.SECTION += 1
        Printer.SUBSECTION = -1
        section_str = Fore.CYAN + f"[{cls.SECTION}] :: "
        print(section_str, *args, **kwargs)
        print(Style.RESET_ALL, end="\n")

    @classmethod
    def subsection(cls, *args, **kwargs):
        Printer.SUBSECTION += 1
        subsection_str = Fore.RED + f"({cls.SUBSECTION}) : "
        print(subsection_str, *args, **kwargs)
        print(Style.RESET_ALL, end="\n")


pprint = Printer.print
section = Printer.section
subsection = Printer.subsection


def to_contest_csv(y_test, name):
    import os
    import numpy as np

    assert np.issubdtype(
        y_test.dtype, np.integer
    ), f"Need integer output for y_test, received : {y_test.dtype}"

    name = name if name.endswith(".csv") else name + ".csv"
    fname = os.path.join(os.path.dirname(__file__), "results", name)

    with open(fname, "w+") as fd:
        fd.write(",Predicition\n")
        for idx, value in enumerate(y_test):
            fd.write(f"{idx},{value}\n")

    return


def bcr(y_true, y_pred):
    def recall(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return tp / (possible_positives + K.epsilon())

    def specificity(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())

    return 0.5 * (recall(y_true, y_pred) + specificity(y_true, y_pred))


def p1(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    m = m1(y_true, y_pred)
    return tp / (m + K.epsilon())


def p2(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    m = m2(y_true, y_pred)
    return tn / (m + K.epsilon())


def m1(y_true, y_pred):
    return K.sum(K.round(y_true))


def m2(y_true, y_pred):
    return K.sum(K.round(K.clip(1 - y_true, 0, 1)))


def P(bcr, bcr_hat, p1, p2, m1, m2):
    bcr_delta = np.abs(bcr - bcr_hat)
    sigma = 0.5 * np.sqrt(((1 - p1) * p1 / m1) + (p2 * (1 - p2) / m2))
    return bcr - bcr_delta * (1 - np.exp(-bcr_delta / sigma))


def P_model(model, X_train, y_train, X_val, y_val, with_info=False):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    bcr_ = bcr(y_train, y_train_pred)
    bcr_hat = bcr(y_val, y_val_pred)

    p1_ = p1(y_val, y_val_pred)
    p2_ = p2(y_val, y_val_pred)
    m1_ = m1(y_val, y_val_pred)
    m2_ = m2(y_val, y_val_pred)

    p = P(bcr_, bcr_hat, p1_, p2_, m1_, m2_)

    if with_info:
        return (
            p,
            {
                "bcr": bcr_,
                "bcr_hat": bcr_hat,
                "p1": p1_,
                "p2": p2_,
                "m1": m1_,
                "m2": m2_,
                "y_train_pred": y_train_pred,
                "y_val_pred": y_val_pred,
            },
        )

    return p


class BCR(tf.keras.metrics.Metric):
    def __init__(self, name="BCR", dtype=None):
        super().__init__(name, dtype=dtype)
        self.TP = tf.keras.metrics.TruePositives()
        self.TN = tf.keras.metrics.TrueNegatives()
        self.FP = tf.keras.metrics.FalsePositives()
        self.FN = tf.keras.metrics.FalseNegatives()

        self.bcr = self.add_weight(name="bcr", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        self.TP.update_state(y_true, y_pred)
        self.TN.update_state(y_true, y_pred)
        self.FP.update_state(y_true, y_pred)
        self.FN.update_state(y_true, y_pred)

        tp = self.TP.result()
        tn = self.TN.result()
        fp = self.FP.result()
        fn = self.FN.result()

        self.bcr.assign(0.5 * ((tp / (tp + fn)) + (tn / (fp + tn))))

    def result(self):
        return self.bcr

    def reset_states(self):
        self.bcr.assign(0.0)


class BCREarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0, restore_best_weights=False):
        super(BCREarlyStopping, self).__init__()
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        bcr = logs.get("bcr")
        val_bcr = logs.get("val_bcr")
        p1 = logs.get("val_p1")
        p2 = logs.get("val_p2")
        m1 = logs.get("val_m1")
        m2 = logs.get("val_m2")
        current = P(bcr, val_bcr, p1, p2, m1, m2)

        if np.less(self.best, current):
            print(f"New best p value : {current}")
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    print(
                        f"Restoring model weights from the end of the best epoch. Best value : {self.best:.3f}"
                    )
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class Report:
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test=None, name=None):
        self._ds = (X_train, y_train, X_val, y_val)
        self._X_test = X_test
        self._model = model
        self._name = name or self._model.__class__.__name__

        self._computed = False
        self.info, self.p = "Not computed", "Not computed"

    def to_contest_csv(self, name=None, y_test=None, save_model=False):
        assert (
            y_test is not None or self._X_test is not None
        ), "Need the X_test or y_test values"

        name = name or self._name

        if y_test is None:
            y_test = self._model.predict(self._X_test)

        to_contest_csv(y_test, name)

        if save_model:
            import tensorflow as tf

            if isinstance(self._model, tf.keras.models.Model):
                model_name = name + ".h5"
                path = os.path.join(os.path.dirname(__file__), "models", model_name)
                self._model.save(path)
            else:
                model_name = name + ".pickle"
                path = os.path.join(os.path.dirname(__file__), "models", model_name)
                pickle.dump(self._model, open(path, "wb+"))

    def compute_performance(self):
        self.p, self.info = P_model(self._model, *self._ds, with_info=True)
        self._computed = True

        return self

    def to_stdout(self):
        if not self._computed:
            self.compute_performance()

        print(f" Report for {self._name} ".center(60, "-"), end="\n" * 2)
        print(f"P score : {self.p.numpy():.3f}")
        print(f'BCR     : {self.info["bcr"].numpy():.3f}')
        print(f'BCR hat : {self.info["bcr_hat"].numpy():.3f}')

        return self


class KFoldTrainingSequential:
    @classmethod
    def compute(
        self,
        X,
        y,
        get_model,
        epochs,
        batch_size,
        callbacks=[],
        k=10,
        shuffle=False,
        seed=None,
    ):
        from sklearn.model_selection import KFold, StratifiedKFold
        from tensorflow.keras.models import clone_model

        kf = StratifiedKFold(k, shuffle=shuffle, random_state=seed)

        results = dict()

        for kidx, (train_indices, val_indices) in enumerate(kf.split(X, y)):
            print(f" Begin fit on fold {kidx+1}/{k} ".center(75, "="), end="\n" * 2)

            result = dict()
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            model = get_model()

            fitted = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            result["model"] = model
            result["fitted"] = fitted
            result["report"] = Report(model, X_train, y_train, X_val, y_val)

            result["report"].to_stdout()

            results[kidx] = result

        return results
