#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a model for predicting polarity of a tweet using
variational inference.

Attributes:
  VarInfAnalyzer (class): class for predicting polarity of a tweet using
    variational inference

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from copy import deepcopy
import numpy as np
import pyro
from pyro.infer import MCMC, NUTS, Predictive
import pyro.poutine as poutine
import torch.nn as nn
import torch

from .alpha import AlphaModel
# from ..constants import CONTRASTIVE_RELS
from ..utils import LOGGER


##################################################################
# Constants
TEST_EPOCHS = 40


##################################################################
# Methods
def check_rel(rel, rel_set):
    """Check whether given relation is present in contrastive rel set.

    Args:
      rel (str or tuple): relation to check
      rel_set (set[str]): relation to check

    Returns:
      bool: true if the given relation is present in contrastive rels

    """
    if rel is None:
        return False
    if isinstance(rel, tuple):
        return check_rel(rel[0], rel_set)
    if isinstance(rel, str):
        return rel.lower() in rel_set
    raise TypeError("Unsupported type {} of relation {!r}.".format(
        type(rel), rel))


##################################################################
# Class
class RDModel(nn.Module):
    """Probabilistic NN model for predicting text's sentiment.

    """
    def __init__(self, rels, n_polarities=3):
        """Class constructor.

        Args:
          rels (set): set of relation tuples
          n_polarities (int): number of polarities to predict

        """
        super().__init__()
        # initialize mapping from relations to indices
        self._logger = LOGGER
        self._rel2idx = {rel_i: i for i, rel_i in enumerate(rels, 1)}
        self._rel2idx[None] = 0
        self.n_rels = len(self._rel2idx)
        self.n_polarities = n_polarities
        # setup testing variables
        self._test_epochs = TEST_EPOCHS
        self._max_test_instances = 10
        self._test_wbench = np.empty((self._test_epochs,
                                      self._max_test_instances,
                                      self.n_polarities))
        # initialize internal models
        self.model = AlphaModel(self.n_rels, self.n_polarities)
        nuts = NUTS(self.model, adapt_step_size=True)
        self._mcmc = MCMC(nuts, num_samples=500, warmup_steps=300)
        self._param_store = pyro.get_param_store()
        self._best_params = None

    @property
    def rel2idx(self):
        return self._rel2idx

    @property
    def best_params(self):
        return self._best_params

    # the model: p(x, z) = p(x|z)p(z)
    def step(self, data):
        """Perform a training step in a single epoch.

        Args:
          data (torch.utils.data.DataLoader): training dataset

        Returns:
          float: loss

        """
        print("data:", data)
        for batch in data:
            self._mcmc.run(*batch)
        return 0.

    def loss(self, x):
        """Evaluate the loss function on the given data.

        Args:
          x (tuple[tensors]): tensors with input data

        """
        return self._svi.evaluate_loss(*x)

    def predict(self, x, trg_y):
        """Predict labels.

        Args:
          x (torch.utils.data.DataLoader):
          trg_y (np.array): array for storing the predicted labels

        Returns:
          float: loss

        """
        # resize `self._test_wbench` if necessary
        n_instances = x[0].shape[0]
        self._resize_wbench(n_instances)
        self._test_wbench *= 0
        with poutine.block():
            with torch.no_grad():
                for wbench_i in self._test_wbench:
                    wbench_i[:n_instances] = self.guide(*x)
        mean = np.mean(self._test_wbench, axis=0)
        trg_y[:] = np.argmax(mean[:n_instances], axis=-1)
        return trg_y

    def debug(self, x, trg_y):
        """Predict labels.

        Args:
          x (torch.utils.data.DataLoader):
          trg_y (np.array): array for storing the predicted labels

        Returns:
          float: loss

        """
        # resize `self._test_wbench` if necessary
        n_instances = x[0].shape[0]
        self._resize_wbench(n_instances)
        self._test_wbench *= 0
        with poutine.block():
            with torch.no_grad():
                for wbench_i in self._test_wbench:
                    wbench_i[:n_instances] = self.guide(*x)
                    break
        self._logger.debug("self._test_wbench: %r",
                           self._test_wbench)
        mean = np.mean(self._test_wbench, axis=0)
        self._logger.debug("self._test_wbench (mean): %r", mean)
        trg_y[:] = np.argmax(mean[:n_instances], axis=-1)
        return trg_y

    def inspect_state(self):
        """Output current pyro parameters.

        """
        for name in self._param_store.get_all_param_names():
            self._logger.info("Param [%s]: %r", name,
                              pyro.param(name).data.numpy())

    def remember_state(self):
        """Remember current pyro parameters.

        """
        self._best_params = deepcopy(self._param_store.get_state())

    def set_state(self, params):
        """Set current pyro parameters.

        """
        self._param_store.set_state(self.best_params)

    def _resize_wbench(self, n_instances):
        if n_instances > self._max_test_instances:
            self._max_test_instances = n_instances
            self._test_wbench = np.resize(
                self._test_wbench,
                (self._test_epochs, self._max_test_instances,
                 self.n_polarities)
            )

    def _reset(self):
        """Remove members which cannot be serialized.

        """
        self._logger.info("Parameters before saving.")
        self.inspect_state()
        self._param_store = None
        self._logger = None

    def _restore(self):
        """Remove members which cannot be serialized.

        """
        self._logger = LOGGER
        self._param_store = pyro.get_param_store()
        self.set_state(self.best_params)
