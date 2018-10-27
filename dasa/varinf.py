#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet using
variational inference.

Attributes:
  VarInfAnalyzer (class): class for predicting polarity of a tweet using
    variational inference

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from builtins import range
from copy import deepcopy
from datetime import datetime
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import RMSprop
from sklearn.metrics import f1_score
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch.nn as nn
import torch

from .alpha import AlphaGuide, AlphaModel
from .dl import N_EPOCHS, OPTIM_PARAM
from .r2n2 import R2N2Analyzer


##################################################################
# Class
class VarInfModel(nn.Module):
    """Network for predicting tweet sentiment using variational inference.

    """

    def __init__(self, rels, n_polarities=3):
        """Class constructor.

        Args:
          rels (set): set of relation tuples
          n_polarities (int): number of polarities to predict

        """
        super(VarInfModel, self).__init__()
        # initialize mapping from relations to indices
        self._rel2idx = {rel_i: i for i, rel_i in enumerate(rels, 1)}
        self._rel2idx[None] = 0
        # initialize internal model
        self.alpha_model = AlphaModel(self._rel2idx)
        self.alpha_guide = AlphaGuide(self._rel2idx)
        self._svi = SVI(self.model, self.guide, optim=RMSprop(OPTIM_PARAM),
                        loss=Trace_ELBO())
        self._test_epochs = 10
        self._max_test_instances = 10
        self.n_polarities = n_polarities
        self._test_wbench = np.empty((self._test_epochs,
                                      self._max_test_instances,
                                      self.n_polarities))

    @property
    def rel2idx(self):
        return self._rel2idx

    # the model: p(x, z) = p(x|z)p(z)
    def model(self, node_scores, children, rels, labels):
        # since we will modify `node_scores` in-place, we would like to
        # preserver the original variant of it to pass it safely to model
        node_scores = deepcopy(node_scores)
        # process minibatch
        n_instances = node_scores.shape[0]
        max_t = node_scores.shape[1]
        max_children = children.shape[-1]
        pyro.module("alpha_model [m]:", self.alpha_model)
        # iterate over each instance of the batch
        with pyro.iarange("batch", size=n_instances) as inst_indices:
            # iterate over each node of the tree in the bottom-up fashion
            for i in range(max_t):
                prnt_scores_i = node_scores[inst_indices, i]
                rels_i = rels[inst_indices, i]
                child_scores_i = self._get_child_scores(
                    node_scores, children, inst_indices, i,
                    n_instances, max_children
                )
                # iterate over each child of that node
                for j in range(max_children):
                    child_scores_ij = child_scores_i[inst_indices, j]
                    var_sfx = "{}_{}".format(i, j)
                    copy_indices, probs2copy, alpha_indices, alpha = \
                        self.alpha_model(
                            var_sfx, prnt_scores_i,
                            child_scores_ij, rels_i[inst_indices, j]
                        )
                    if probs2copy is not None:
                        node_scores[inst_indices[copy_indices], i] = probs2copy
                    if alpha is not None:
                        z_ij = pyro.sample(
                            "z_{}_{}".format(i, j), dist.Dirichlet(alpha))
                        node_scores[inst_indices[alpha_indices], i] = z_ij
            return pyro.sample("y", dist.Categorical(z_ij),
                               obs=labels[inst_indices])

    # the guide (i.e., variational distribution): q(z|x)
    def guide(self, node_scores, children, rels, labels):
        # since we will modify `node_scores` in-place, we would like to
        # preserver the original variant of it to pass it safely to model
        node_scores = deepcopy(node_scores)
        # process minibatch
        n_instances = node_scores.shape[0]
        max_t = node_scores.shape[1]
        max_children = children.shape[-1]
        pyro.module("alpha_guide", self.alpha_guide)
        # iterate over each instance of the batch
        with pyro.iarange("batch", size=n_instances) as inst_indices:
            # iterate over each node of the tree in the bottom-up fashion
            for i in range(max_t):
                prnt_scores_i = node_scores[inst_indices, i]
                rels_i = rels[inst_indices, i]
                child_scores_i = self._get_child_scores(
                    node_scores, children, inst_indices, i,
                    n_instances, max_children
                )
                # iterate over each child of that node
                for j in range(max_children):
                    child_scores_ij = child_scores_i[inst_indices, j]
                    var_sfx = "{}_{}".format(i, j)
                    copy_indices, probs2copy, alpha_indices, alpha = \
                        self.alpha_model(
                            var_sfx, prnt_scores_i,
                            child_scores_ij, rels_i[inst_indices, j]
                        )
                    if probs2copy is not None:
                        node_scores[inst_indices[copy_indices], i] = probs2copy
                    if alpha is not None:
                        z_ij = pyro.sample(
                            "z_{}_{}".format(i, j), dist.Dirichlet(alpha))
                        node_scores[inst_indices[alpha_indices], i] = z_ij
            return node_scores[inst_indices, -1]

    def _get_child_scores(self, node_scores, children, inst_indices, i,
                          n_instances, max_children):
        child_indices = children[inst_indices, i].reshape(-1)
        inst_indices.repeat(max_children, 1).t().reshape(-1)
        child_scores = node_scores[
            inst_indices.repeat(max_children, 1).t().reshape(-1),
            child_indices
        ].reshape(n_instances, max_children, -1)
        return child_scores

    def step(self, data):
        """Perform a training step on a single epoch.

        Args:
          data (torch.utils.data.DataLoader): training dataset

        Returns:
          float: loss

        """
        loss = 0.
        for batch_j in data:
            loss += self._svi.step(*batch_j)
        return loss

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
        # print("self._test_wbench:", repr(self._test_wbench))
        with poutine.block():
            with torch.no_grad():
                for wbench_i in self._test_wbench:
                    wbench_i[:n_instances] = self.guide(*x)
        # print("* self._test_wbench:", repr(self._test_wbench))
        mean = np.mean(self._test_wbench, axis=0)
        # print("predict trg_y: ", repr(trg_y))
        # print("predict mean: ", repr(mean[:n_instances]))
        trg_y[:] = np.argmax(mean[:n_instances], axis=-1)
        # print("evaluate labels: ", repr(labels))
        return trg_y

    def loss(self, x):
        """Evaluate the loss function on the given data.

        Args:
          x (tuple[tensors]): tensors with input data

        """
        return self._svi.evaluate_loss(*x)

    def _resize_wbench(self, n_instances):
        if n_instances > self._max_test_instances:
            self._max_test_instances = n_instances
            self._test_wbench = np.resize(
                self._test_wbench,
                (self._test_epochs, self._max_test_instances,
                 self.n_polarities)
            )


class VarInfAnalyzer(R2N2Analyzer):
    """Discourse-aware sentiment analysis using variational inference.

    Attributes:

    """

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(VarInfAnalyzer, self).__init__(*args, **kwargs)
        self._name = "VarInf"

    def predict(self, instances):
        raise NotImplementedError

    def debug(self, instance):
        raise NotImplementedError

    def _init_model(self, forrest):
        """Initialize the model that will be used for prediction.

        Args:
          forrest (list[RSTTree]): list of all training RST trees

        """
        rels = self.get_rels(forrest)
        self._model = VarInfModel(rels)

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
        best_model = None
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
                best_model = deepcopy(self._model.state_dict())
            self._logger.info(
                "Epoch %d finished in %d sec [train loss: %f, "
                "train macro-F1: %f, dev loss: %f, dev macro-F1: %f]%s",
                epoch_i, (epoch_end - epoch_start).total_seconds(),
                train_loss, train_macro_f1, dev_loss, dev_macro_f1,
                "*" if selected else ""
            )
        self._model.load_state_dict(best_model)
        self._logger.debug("Model trained...")
        return best_f1

    def _digitize_data(self, data, train_mode=False):
        dataloader = super(VarInfAnalyzer, self)._digitize_data(
            data, train_mode
        )
        # remove message scores as they will be incorporated as regular priors
        # of the root node
        tensors = list(dataloader.dataset.tensors)
        tensors.pop(-2)
        dataloader.dataset.tensors = tuple(tensors)
        return dataloader

    def tree2mtx(self, node_scores, children, rels, tree, instance):
        super(VarInfAnalyzer, self).tree2mtx(node_scores, children, rels,
                                             tree, instance)
        assert np.sum(node_scores[-1, :]) == 0, \
            "Scores of the root node are not equal 0."
        node_scores[-1, :] = instance["polarity_scores"]
