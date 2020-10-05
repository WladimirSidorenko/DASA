#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing classes for guessing the alpha scores of a Dirichlet
distribution based on the polarities of parent and child.

Attributes:
  AlphaModel (class): class for guessing the alpha value in a pyro model
  AlphaGuide (class): class for guessing the alpha value in a pyro guide

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from pyro.nn.module import PyroModule, PyroParam, PyroSample
from torch import tensor
from pyro.distributions import (
    constraints, Beta, Chi2, Categorical, Dirichlet, MultivariateNormal
)
from torch.nn import Softmax
import numpy as np
import pyro
import torch


##################################################################
# Classes
class AlphaModel(PyroModule):
    def __init__(self, n_rels, n_polarities=3):
        super().__init__()
        self._n_rels = n_rels
        self._n_polarities = n_polarities
        self._min_sum = tensor(1e-10)
        self._softmax = Softmax(dim=-1)

    @PyroParam
    def M_Mu(self):
        Mu = np.eye(self._n_polarities, dtype="float32")
        Mu[1, 1] = 0.3
        Mu = np.tile(Mu, (self._n_rels, 1)).reshape(
            self._n_rels, self._n_polarities, 1, self._n_polarities)
        return tensor(Mu)

    @PyroParam
    def M_Sigma(self):
        Sigma = np.eye(self._n_polarities, dtype="float32")
        return tensor(Sigma)

    @PyroSample
    def M(self):
        return MultivariateNormal(self.M_Mu, self.M_Sigma)

    @PyroParam
    def beta_p(self):
        return 5. * tensor(np.ones((self._n_rels, self._n_polarities, 1),
                                   dtype="float32"))

    @PyroParam
    def beta_q(self):
        return 5. * tensor(np.ones((self._n_rels, self._n_polarities, 1),
                                   dtype="float32"))

    @PyroSample
    def beta(self):
        return Beta(self.beta_p, self.beta_q)

    @PyroParam(constraint=constraints.positive)
    def _scale_factor(self):
        return tensor(34.)

    @PyroSample
    def scale_factor(self):
        return Chi2(self._scale_factor)

    @PyroParam(constraint=constraints.positive)
    def z_epsilon(self):
        return tensor(1e-2)

    def forward(self, node_scores, children, rels, labels):
        """Perform inference on a single batch.

        """
        # work on a copy of node scores in order not to affect the original
        # training set
        node_scores = node_scores.clone()
        # figure-out batch dimensions
        # number of instances
        n_instances = node_scores.shape[0]
        # maximum tree depth
        max_depth = node_scores.shape[1]
        # maximum tree width
        max_width = children.shape[-1]
        # iterate over each instance of the batch
        with pyro.plate("batch", size=n_instances) as inst_indices:
            print("inst_indices:", inst_indices)
            # iterate over each node of the tree in the bottom-up fashion
            for i in range(max_depth):
                prnt_scores_i = node_scores[inst_indices, i]
                rels_i = rels[inst_indices, i]
                child_scores_i = self._get_child_scores(
                    node_scores, children, inst_indices, i,
                    n_instances, max_width
                )
                for j in range(max_width):
                    child_scores_ij = child_scores_i[inst_indices, j]
                    var_sfx = "{}_{}".format(i, j)
                    copy_indices, probs2copy, alpha_indices, alpha = \
                        self._forward_node(
                            var_sfx, prnt_scores_i,
                            child_scores_ij, rels_i[inst_indices, j]
                        )
                    if probs2copy is not None:
                        node_scores[inst_indices[copy_indices], i] = probs2copy
                    if alpha is not None:
                        z_ij = pyro.sample(
                            "z_{}_{}".format(i, j), Dirichlet(alpha))
                        node_scores[inst_indices[alpha_indices], i] = z_ij
                        prnt_scores_i = node_scores[inst_indices, i]
            z_ij = node_scores[inst_indices, -1]
            y = pyro.sample("y", Categorical(z_ij), obs=labels[inst_indices])
        return y

    def _forward_node(self, var_sfx: str, prnt_probs: tensor,
                      child_probs, rels):
        """Update probability scores for a single node in the discourse tree.

        Args:
          var_sfx (str): suffix of variable names (corresponds to node index in
            the tree)

        """
        # The vector `nz_chld_indices` will contain indices of the child_probs
        # which are not zero.
        nz_chld_indices = child_probs.sum(dim=-1).nonzero().squeeze(-1)
        if nz_chld_indices.nelement() == 0:
            return None, None, None, None
        # Only leave `parent`, `child`, and `rels` elements for which
        # `child_probs` are not zero.
        child_probs = child_probs[nz_chld_indices].unsqueeze(1)
        rels = rels[nz_chld_indices]
        M_nz_rels = self.M[rels, :, nz_chld_indices, :]
        M_nz_rels = M_nz_rels.reshape(
            -1, self._n_polarities, self._n_polarities
        )
        # Batch-multiply the remaining child scores with relation matrices (M).
        child_probs = torch.bmm(child_probs, M_nz_rels)
        # Normalize child probabilities:
        child_probs = self._softmax(child_probs)
        # Custom Normalization
        # We will only take a convex combination of the means if the parent
        # probs are also non-zero, otherwise we will assign the child probs to
        # the parents unchanged.
        prnt_probs = prnt_probs[nz_chld_indices]
        prnt_probs_sum = prnt_probs.sum(dim=-1)
        z_prnt_indices = (prnt_probs_sum == 0.).nonzero().squeeze_(-1)
        # indices of instances whose child scores are non-zero, but parent
        # scores are zero
        copy_indices = nz_chld_indices[z_prnt_indices]
        child_probs = torch.squeeze(child_probs, 1)
        child_probs2copy = child_probs[z_prnt_indices]
        nz_prnt_indices = prnt_probs_sum.nonzero().squeeze(-1)
        alpha_indices = nz_chld_indices[nz_prnt_indices]
        if alpha_indices.nelement() == 0:
            alpha = None
        else:
            child_probs = child_probs[nz_prnt_indices]
            prnt_probs = prnt_probs[nz_prnt_indices]
            rels = rels[nz_prnt_indices]
            # take a convex combination of the parent and child scores as new
            # values for alpha and scale the resulting alpha scores with the
            # corresponding scale factor.
            beta = self.beta[rels, :, alpha_indices]
            alpha = (1. - beta) * prnt_probs + beta * child_probs
            scale = self.scale(
                prnt_probs, child_probs, alpha_indices
            ).unsqueeze_(-1)
            alpha *= scale
        if torch.isnan(child_probs2copy).any():
            print("alpha:", alpha)
            print("M:", self.__class__.__name__, self.M)
            print("nz_chld_indices:", nz_chld_indices)
            print("child_probs:", child_probs)
            print("prnt_probs:", prnt_probs)
            print("child_probs2copy:", child_probs2copy)
            exit(66)
        return copy_indices, child_probs2copy, alpha_indices, alpha

    def _get_child_scores(self, node_scores, children, inst_indices, i,
                          n_instances, max_children):
        child_indices = torch.flatten(children[inst_indices, i])
        child_scores = node_scores[
            torch.flatten(inst_indices.repeat(max_children, 1).t()),
            child_indices
        ].reshape(n_instances, max_children, -1)
        return child_scores

    def scale(self, prnt_probs, child_probs, alpha_indices):
        z = torch.clamp(prnt_probs + child_probs, min=self.z_epsilon.item())
        z_norm = 1. / torch.sum(z, dim=-1)
        z_norm.unsqueeze_(-1)
        z *= z_norm
        entropy = -torch.sum(z * torch.log(z), dim=-1)
        cos = torch.sum(prnt_probs * child_probs, dim=-1)
        norm = torch.norm(prnt_probs, dim=-1) * torch.norm(child_probs, dim=-1)
        # replace 0's with 1's to prevent division by 0, since cosine in
        # this case will be 0 anyway
        norm = torch.clamp(norm, min=self._min_sum)
        cos = 0.1 + cos / norm
        scale_factor = self.scale_factor[alpha_indices]
        scale = scale_factor * cos / entropy
        return scale
