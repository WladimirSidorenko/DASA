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
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import RMSprop
from six import iteritems, string_types
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch.nn as nn
import torch

from .alpha import AlphaGuide, AlphaModel
from ..constants import CONTRASTIVE_RELS
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
        self.softplus = nn.Softplus()
        # initialize internal models
        self.alpha_model = AlphaModel(self.n_rels)
        self._alpha_model_priors = None
        self.alpha_guide = AlphaGuide(self.n_rels)
        self._alpha_guide_prior_params = None
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
        # preserver the original variant of it to pass it safely to model
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
        priors = self._get_guide_priors()
        alpha_guide = pyro.random_module("alpha_guide", self.alpha_guide,
                                         priors)()
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
                        alpha_guide(
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
        # print("self._test_wbench:", repr(self._test_wbench))
        with poutine.block():
            with torch.no_grad():
                for wbench_i in self._test_wbench:
                    wbench_i[:n_instances] = self.guide(*x)
        mean = np.mean(self._test_wbench, axis=0)
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
        M_mu[1, :] = [-0.25, 0.5, -0.25]
        M_mu = np.tile(M_mu, (self.n_rels, 1)).reshape(
            self.n_rels, self.n_polarities, self.n_polarities
        )
        for rel, rel_idx in iteritems(self.rel2idx):
            # swap axes for contrastive relations
            if check_rel(rel, CONTRASTIVE_RELS):
                mu_i = M_mu[rel_idx]
                mu_i[[0, 2]] = mu_i[[2, 0]]
        M_mu = torch.tensor(M_mu)
        M_sigma = torch.tensor(
            np.ones((self.n_rels, self.n_polarities, self.n_polarities),
                    dtype="float32")
        )
        # beta
        beta_p = 15. * torch.tensor(np.ones((self.n_rels, 1),
                                            dtype="float32"))
        beta_q = 15. * torch.tensor(np.ones((self.n_rels, 1),
                                            dtype="float32"))
        # z_epsilon
        z_epsilon_p = torch.tensor(1.)
        z_epsilon_q = torch.tensor(15.)
        # scale factor
        scale_factor = torch.tensor(42.)
        return {"M_mu": M_mu, "M_sigma": M_sigma, "beta_p": beta_p,
                "beta_q": beta_q, "z_epsilon_p": z_epsilon_p,
                "z_epsilon_q": z_epsilon_q, "scale_factor": scale_factor}

    def _get_model_priors(self):
        """Initialize priors for alpha model.

        Returns:
          dict: dictionary of priors

        """
        if self._alpha_model_priors:
            return self._alpha_model_priors
        # sample the variables from their corresponding distributions
        params = self._get_prior_params()
        self._alpha_model_priors = self._params2probs(params)
        return self._alpha_model_priors

    def _get_guide_priors(self):
        """Initialize priors for alpha guide.

        Args:
          guide_mode (bool): create priors for guide (i.e., wrap relevant
            parameters into `pyro.param`)

        Returns:
          dict: dictionary of priors

        """
        if not self._alpha_guide_prior_params:
            # create initial parameters
            params = self._get_prior_params()
            # register all parameters in pyro
            for p, v in iteritems(params):
                pyro.param(p, v)
            self._alpha_guide_prior_params = dict(
                self._param_store.named_parameters()
            )
        else:
            # register all parameters in pyro
            for p, v in iteritems(self._alpha_guide_prior_params):
                pyro.param(p, v)
        return self._params2probs(self._alpha_guide_prior_params)

    def _params2probs(self, params):
        """Convert parameters to probability distributions.

        Args:
          params (dict[str -> np.array]): dictionary of distribution parameters

        Returns:
          dict[str -> Dist]: dictionary of prior probabilities

        """
        M = dist.Normal(params["M_mu"],
                        self.softplus(params["M_sigma"])).independent(2)
        beta = dist.Beta(self.softplus(params["beta_p"]),
                         self.softplus(params["beta_q"])).independent(1)
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
