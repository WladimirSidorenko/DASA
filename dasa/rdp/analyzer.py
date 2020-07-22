#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet using
recursive Dirichlet process.

Attributes:
  RDPAnalyzer (class): class for predicting polarity of a tweet using
    a recursive Dirichlet process

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from builtins import range
from datetime import datetime
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from typing import List, Optional
import numpy as np
import pyro
import torch

from .model import RDPModel
from ..constants import IDX2CLS
from ..dl import N_EPOCHS
from ..r2n2 import R2N2Analyzer


##################################################################
# Setup
torch.set_printoptions(precision=8)


##################################################################
# Class
class RDPAnalyzer(R2N2Analyzer):
    """Discourse-aware sentiment analysis using variational inference.

    Attributes:

    """
    _name = "RDP"

    def __init__(self, relation_scheme: str, sentiment_scores: str,
                 n_classes: int):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super().__init__(relation_scheme, sentiment_scores, n_classes)
        self._wbench_y = None

    def predict_instance(self, instance: dict,
                         relation_scheme: Optional[str] = None,
                         sentiment_scores: Optional[str] = None) -> str:
        tree = self.span2nuc(
            self.build_rst(instance, relation_scheme, sentiment_scores)
        )
        self._resize(tree)
        self.tree2mtx(self._wbench_node_scores[0, :],
                      self._wbench_children[0, :],
                      self._wbench_rels[0, :],
                      tree, instance, sentiment_scores)
        self._model.predict([torch.from_numpy(self._wbench_node_scores),
                             torch.from_numpy(self._wbench_children),
                             torch.from_numpy(self._wbench_rels),
                             None],
                            self._wbench_y)
        return IDX2CLS[self._wbench_y[0]]

    def debug(self, instance: dict,
              relation_scheme: Optional[str] = None,
              sentiment_scores: Optional[str] = None) -> str:
        tree = self.span2nuc(
            self.build_rst(instance, relation_scheme, sentiment_scores)
        )
        self._resize(tree)
        self._logger.debug("tree: %r", tree)
        self.tree2mtx(self._wbench_node_scores[0, :],
                      self._wbench_children[0, :],
                      self._wbench_rels[0, :],
                      tree, instance, sentiment_scores)
        self._logger.debug("node_scores: %r", self._wbench_node_scores)
        self._logger.debug("children: %r", self._wbench_children)
        self._logger.debug("rels: %r", self._wbench_rels)
        self._model.debug([torch.from_numpy(self._wbench_node_scores),
                           torch.from_numpy(self._wbench_children),
                           torch.from_numpy(self._wbench_rels),
                           None],
                          self._wbench_y)
        self._logger.debug("prediction: %r", self._wbench_y)
        pred = IDX2CLS[self._wbench_y[0]]
        self._logger.debug("predicted class: %r", pred)
        return pred

    def _init_model(self, forrest):
        """Initialize the model that will be used for prediction.

        Args:
          forrest (list[RSTTree]): list of all training RST trees

        """
        rels = self.get_rels(forrest)
        self._model = RDPModel(rels)

    def _init_wbenches(self):
        super()._init_wbenches()
        self._wbench_y = np.zeros(1, dtype="long")

    def _train(self, train_set, dev_set):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list): training set
          dev_set (list): development set

        Returns:
          float: best macro-averaged F1 observed on the dev set

        """
        self._logger.debug("Training model...")
        # prepare matrices for storing gold and predicted labels of the
        # training and development sets
        X_train = train_set.dataset.tensors
        Y_train = np.empty((2, len(train_set.dataset)), dtype="int32")
        Y_train[0, :] = X_train[-1]
        X_dev = dev_set.dataset.tensors
        Y_dev = np.empty((2, len(dev_set.dataset)), dtype="int32")
        Y_dev[0, :] = X_dev[-1]
        # optimize model on the training set
        best_f1 = -1.
        best_dev_loss = np.Inf
        pyro.clear_param_store()
        for epoch_i in range(N_EPOCHS):
            selected = False
            epoch_start = datetime.utcnow()
            train_loss = self._model.step(train_set)
            self._model.predict(X_train, Y_train[1])
            train_macro_f1 = f1_score(Y_train[0], Y_train[1], average="macro")
            dev_loss = self._model.loss(X_dev)
            self._model.predict(X_dev, Y_dev[1])
            dev_macro_f1 = f1_score(Y_dev[0], Y_dev[1], average="macro")
            epoch_end = datetime.utcnow()
            if best_f1 < dev_macro_f1 or (best_f1 == dev_macro_f1
                                          and dev_loss < best_dev_loss):
                best_f1 = dev_macro_f1
                best_dev_loss = dev_loss
                selected = True
                self._model.remember_state()
            self._logger.info(
                "Epoch %d finished in %d sec [train loss: %f, "
                "train macro-F1: %f, dev loss: %f, dev macro-F1: %f]%s",
                epoch_i, (epoch_end - epoch_start).total_seconds(),
                train_loss, train_macro_f1, dev_loss, dev_macro_f1,
                "*" if selected else ""
            )
        self._logger.info("Model trained...")
        self._model.set_state(self._model.best_params)
        self._logger.info("Best parameters...")
        self._model.inspect_state()
        return best_f1

    def _digitize_data(self, X: List[dict], Y: np.array,
                       train_mode: bool = False) -> DataLoader:
        dataloader = super()._digitize_data(X, Y, train_mode)
        # remove message scores as they will be incorporated as regular priors
        # of the root node
        tensors = list(dataloader.dataset.tensors)
        tensors.pop(-2)
        dataloader.dataset.tensors = tuple(tensors)
        return dataloader

    def tree2mtx(self, node_scores, children, rels, tree,
                 instance, sentiment_scores: Optional[str] = None):
        super().tree2mtx(node_scores, children, rels, tree, instance)
        assert np.sum(node_scores[-1, :]) == 0, \
            "Scores of the root node do not equal 0."
        # node_scores[-1, :] = 1. / len(instance["polarity_scores"])
        node_scores[-1, :] = self._get_scores(instance, sentiment_scores)

    def _reset(self):
        """Remove members which cannot be serialized.

        """
        self._model._reset()
        super()._reset()

    def _restore(self, a_path):
        """Remove members which cannot be serialized.

        """
        super()._restore(a_path)
        self._model._restore()
