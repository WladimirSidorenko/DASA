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

from builtins import range
from copy import deepcopy
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import RMSprop
from six import iteritems, string_types
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch.nn as nn
import torch

from .alpha import AlphaModel
# from ..constants import CONTRASTIVE_RELS
from ..dl import OPTIM_PARAM
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
    if isinstance(rel, string_types):
        return rel.lower() in rel_set
    raise TypeError("Unsupported type {} of relation {!r}.".format(
        type(rel), rel))


##################################################################
# Class
class RDPModel(nn.Module):
    """Network for predicting tweet sentiment using variational inference.

    """

    def __init__(self, rels, n_polarities=3):
        """Class constructor.

        Args:
          rels (set): set of relation tuples
          n_polarities (int): number of polarities to predict

        """
        super(RDPModel, self).__init__()
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
        self.alpha_model = AlphaModel(self.n_rels)
        self.alpha_guide = AutoDelta(self.alpha_model)
        self._param_store = pyro.get_param_store()
        self._best_params = None
        self._svi = SVI(self.model, self.guide, optim=RMSprop(OPTIM_PARAM),
                        loss=Trace_ELBO())

    @property
    def rel2idx(self):
        return self._rel2idx

    @property
    def best_params(self):
        return self._best_params

    # the model: p(x, z) = p(x|z)p(z)
    def model(self, node_scores, children, rels, labels):
        # since we will modify `node_scores` in-place, we would like to
        # preserve the original variant of it to pass it safely to model
        node_scores = deepcopy(node_scores)
        # process minibatch
        n_instances = node_scores.shape[0]
        max_t = node_scores.shape[1]
        max_children = children.shape[-1]
        alpha_model = pyro.random_module(
            "alpha_model", self.alpha_model, self._get_model_priors()
        )()
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
                        alpha_model(
                            var_sfx, prnt_scores_i,
                            child_scores_ij, rels_i[inst_indices, j]
                        )
                    if probs2copy is not None:
                        node_scores[inst_indices[copy_indices], i] = probs2copy
                    if alpha is not None:
                        z_ij = pyro.sample(
                            "z_{}_{}".format(i, j), dist.Dirichlet(alpha))
                        node_scores[inst_indices[alpha_indices], i] = z_ij
                        prnt_scores_i = node_scores[inst_indices, i]
            z_ij = node_scores[inst_indices, -1]
            y = pyro.sample("y", dist.Categorical(z_ij),
                            obs=labels[inst_indices])
            return y

    # the guide (i.e., variational distribution): q(z|x)
    def guide(self, node_scores, children, rels, labels):
        # since we will modify `node_scores` in-place, we would like to
        # preserver the original variant of it to pass it safely to model
        node_scores = deepcopy(node_scores)
        # process minibatch
        n_instances = node_scores.shape[0]
        max_t = node_scores.shape[1]
        max_children = children.shape[-1]
        priors = self._get_guide_priors()
        self._logger.debug("guide priors: %r", priors)
        alpha_guide = pyro.random_module("alpha_guide", self.alpha_guide,
                                         priors)()
        self._logger.debug("alpha_guide.z_epsilon: %r", alpha_guide.z_epsilon)
        self._logger.debug("alpha_guide.M: %r", alpha_guide.M)
        self._logger.debug("alpha_guide.beta: %r", alpha_guide.beta)
        self._logger.debug("alpha_guide.scale_factor: %r",
                           alpha_guide.scale_factor)
        # iterate over each instance of the batch
        with pyro.iarange("batch", size=n_instances) as inst_indices:
            # iterate over each node of the tree in the bottom-up fashion
            for i in range(max_t):
                self._logger.debug("Considering time step %d", max_t)
                prnt_scores_i = node_scores[inst_indices, i]
                self._logger.debug("prnt_scores[%d]: %r", max_t, prnt_scores_i)
                rels_i = rels[inst_indices, i]
                self._logger.debug("rels[%d]: %r", max_t, rels_i)
                child_scores_i = self._get_child_scores(
                    node_scores, children, inst_indices, i,
                    n_instances, max_children
                )
                self._logger.debug("child_scores_i: %r", child_scores_i)
                self._logger.debug("child_scores[%d]: %r", max_t, rels_i)
                # iterate over each child of that node
                for j in range(max_children):
                    self._logger.debug("Considering child %d", j)
                    child_scores_ij = child_scores_i[inst_indices, j]
                    var_sfx = "{}_{}".format(i, j)
                    self._logger.debug("sampling variable %s", var_sfx)
                    self._logger.debug("prnt_scores_i: %r", prnt_scores_i)
                    self._logger.debug("child_scores_ij: %r", child_scores_ij)
                    copy_indices, probs2copy, alpha_indices, alpha = \
                        alpha_guide(
                            var_sfx, prnt_scores_i,
                            child_scores_ij, rels_i[inst_indices, j]
                        )
                    self._logger.debug("alpha %r", alpha)
                    self._logger.debug("probs2copy %r", probs2copy)
                    if probs2copy is not None:
                        node_scores[inst_indices[copy_indices], i] = probs2copy
                    if alpha is not None:
                        z_ij = pyro.sample(
                            "z_{}_{}".format(i, j), dist.Dirichlet(alpha))
                        if torch.isnan(z_ij).any() or torch.isinf(z_ij).any():
                            print("z_{}_{}:".format(i, j), z_ij)
                            exit(66)
                        node_scores[inst_indices[alpha_indices], i] = z_ij
                        prnt_scores_i = node_scores[inst_indices, i]
            return node_scores[inst_indices, -1]

    def step(self, data):
        """Perform a training step on a single epoch.

        Args:
          data (torch.utils.data.DataLoader): training dataset

        Returns:
          float: loss

        """
        loss = 0.
        for j, batch_j in enumerate(data):
            print("j:", j)
            loss += self._svi.step(*batch_j)
            print("loss:", loss)
            if (torch.isnan(self.alpha_guide.M.flatten()).any()
                    or torch.isinf(self.alpha_guide.M.flatten()).any()):
                exit(66)
        return loss

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

    def _get_child_scores(self, node_scores, children, inst_indices, i,
                          n_instances, max_children):
        child_indices = children[inst_indices, i].reshape(-1)
        inst_indices.repeat(max_children, 1).t().reshape(-1)
        child_scores = node_scores[
            inst_indices.repeat(max_children, 1).t().reshape(-1),
            child_indices
        ].reshape(n_instances, max_children, -1)
        return child_scores

    def _get_prior_params(self):
        """Initialize priors which are common for model and guide.

        Returns:
          dict[str -> np.array]: dictionary of distribution parameters

        """
        # relation transformation matrix
        M_mu = np.eye(self.n_polarities, dtype="float32")
        M_mu[1, 1] = 0.3
        M_mu = torch.tensor(
            np.tile(M_mu, (self.n_rels, 1)).reshape(
                self.n_rels, self.n_polarities, self.n_polarities
            )
        )
        M_sigma = torch.tensor(np.eye(self.n_polarities, dtype="float32"))
        # beta
        beta_p = 5. * torch.tensor(np.ones((1, self.n_polarities),
                                           dtype="float32"))
        beta_q = 5. * torch.tensor(np.ones((1, self.n_polarities),
                                           dtype="float32"))
        # z_epsilon
        z_epsilon_p = torch.tensor(1.)
        z_epsilon_q = torch.tensor(15.)
        # scale factor
        scale_factor = torch.tensor(34.)
        return {"M_mu": M_mu, "M_sigma": M_sigma, "beta_p": beta_p,
                "beta_q": beta_q, "z_epsilon_p": z_epsilon_p,
                "z_epsilon_q": z_epsilon_q, "scale_factor": scale_factor}

    def _params2probs(self, params):
        """Convert parameters to probability distributions.

        Args:
          params (dict[str -> np.array]): dictionary of distribution parameters

        Returns:
          dict[str -> Dist]: dictionary of prior probabilities

        """
        print(params["M_mu"])
        M = dist.MultivariateNormal(
            params["M_mu"], self.softplus(params["M_sigma"])
        ).independent(2)
        beta = dist.Beta(self.softplus(params["beta_p"]),
                         self.softplus(params["beta_q"])).expand(
                            [self.n_rels, self.n_polarities]
                            ).to_event(2)
        z_epsilon = dist.Beta(
            self.softplus(params["z_epsilon_p"]),
            self.softplus(params["z_epsilon_q"]))
        scale_factor = dist.Chi2(params["scale_factor"])
        return {"M": M, "beta": beta, "z_epsilon": z_epsilon,
                "scale_factor": scale_factor}

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
        self._alpha_guide_prior_params = None
        self._param_store = None
        self._logger = None

    def _restore(self):
        """Remove members which cannot be serialized.

        """
        self._logger = LOGGER
        self._param_store = pyro.get_param_store()
        self.set_state(self.best_params)
        self._alpha_guide_prior_params = dict(
            self._param_store.named_parameters()
        )
