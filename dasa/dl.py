#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing an abstract base for deep-learning--based analyzers.

Attributes:
  DLBaseAnalyzer (class): abstract base for all deep-learning--based analyzers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from builtins import range
from copy import deepcopy
from datetime import datetime
from sklearn.metrics import f1_score
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import DASBaseAnalyzer
from .constants import CLS2IDX

##################################################################
# Variables and Constants
N_EPOCHS = 100
DATALOADER_KWARGS = {
    "batch_size": 16,
    "shuffle": True
}


##################################################################
# Classes
class DLBaseAnalyzer(DASBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(DLBaseAnalyzer, self).__init__(*args, **kwargs)
        self._model = None
        self._name = "DLBase"
        self._train_criterion = nn.MultiMarginLoss(margin=3e-3)
        self._dev_criterion = nn.MultiMarginLoss(margin=3e-3)
        self._optim_cls = optim.RMSprop
        self._n_cls = len(CLS2IDX)
        self._wbench = np.zeros((1, self._n_cls))

    def train(self, train_set, dev_set=None,
              grid_search=True, balance=False):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list or None):
            training set
          dev_set (list or None):
            development set
          grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          balance (bool): balance dataset to get equal number of instances
            for all classes (via downsampling)

        Returns:
          void:

        """
        self._logger.debug("Preparing data...")
        train_set, dev_set = self._prepare_data(train_set, dev_set)
        self._logger.debug("Data prepared...")
        self._logger.debug("Training model...")
        optimizer = self._optim_cls(self._model.parameters())
        # prepare matrices for storing gold and predicted labels on training
        # and test sets
        n_train = len(train_set.dataset)
        Y_train = np.empty((2, n_train), dtype="int32")
        n_dev = len(dev_set.dataset)
        Y_dev = np.empty((2, n_dev), dtype="int32")
        # optimize model on the training set
        best_model = None
        best_f1 = -1.
        for epoch_i in range(N_EPOCHS):
            selected = False
            train_loss = 0.
            dev_loss = 0.
            i = next_i = 0
            epoch_start = datetime.utcnow()
            for train_batch in train_set:
                inputs = train_batch[:-1]
                labels = train_batch[-1]
                optimizer.zero_grad()
                outputs = self._model(*inputs)
                # compute loss and evaluate model's performance on the training
                # data
                loss = self._train_criterion(outputs, labels)
                train_loss += loss.item()
                _, predicted_labels = torch.max(outputs, 1)
                next_i = i + len(predicted_labels)
                Y_train[0, i:next_i] = labels.numpy()
                Y_train[1, i:next_i] = predicted_labels.numpy()
                i = next_i
                # optimize model
                loss.backward()
                optimizer.step()
            # evaluate model's performance on the dev set
            with torch.no_grad():
                j = next_j = 0
                for dev_batch in dev_set:
                    inputs = dev_batch[:-1]
                    labels = dev_batch[-1]
                    outputs = self._model(*inputs)
                    loss = self._dev_criterion(outputs, labels)
                    dev_loss += loss.item()
                    _, predicted_labels = torch.max(outputs, 1)
                    next_j = j + len(predicted_labels)
                    Y_dev[0, j:next_j] = labels.numpy()
                    Y_dev[1, j:next_j] = predicted_labels.numpy()
                    j = next_j
            epoch_end = datetime.utcnow()
            assert i == n_train and j == n_dev, \
                ("The number of predicted labels is less than the number"
                 " of input instances.")
            train_macro_f1 = f1_score(Y_train[0], Y_train[1], average="macro")
            dev_macro_f1 = f1_score(Y_dev[0], Y_dev[1], average="macro")
            if best_f1 < dev_macro_f1:
                best_f1 = dev_macro_f1
                selected = True
                best_model = deepcopy(self._model)
            self._logger.info(
                "Epoch %d finished in %d sec [train loss:"
                " %f, train macro-F1: %f, dev loss: %f, dev macro-F1: %f]%s",
                epoch_i, (epoch_end - epoch_start).total_seconds(),
                train_loss, train_macro_f1, dev_loss, dev_macro_f1,
                "*" if selected else ""
            )
        self._model = best_model
        self._logger.debug("Model trained...")

    def _prepare_data(self, train_set, dev_set):
        """Provide train/test split and digitize the data.

        """
        if not dev_set and len(train_set) > 1:
            n = len(train_set)
            n_dev = max(1, n // 15)
            idcs = list(range(n))
            np.random.shuffle(idcs)

            def get_split(data, idcs):
                return [data[i] for i in idcs]

            dev_set = get_split(train_set, idcs[:n_dev])
            train_set = get_split(train_set, idcs[n_dev:])

        # convert tweets to word indices
        train_set = self._digitize_data(train_set)
        dev_set = self._digitize_data(dev_set)
        return (train_set, dev_set)

    @abc.abstractmethod
    def _digitize_data(self, data):
        raise NotImplementedError
