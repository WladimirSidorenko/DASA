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
from sparsemax import Sparsemax
from torch.distributions import constraints, Beta, MultivariateNormal
from torch.nn import Softmax
import numpy as np
import torch


##################################################################
# Classes
class AlphaModel(PyroModule):
    def __init__(self, n_rels, n_polarities=3):
        super().__init__()
        self._n_rels = n_rels
        self._n_polarities = n_polarities
        self._sparsemax = Sparsemax(dim=-1)
        self._softmax = Softmax(dim=-1)

    @PyroParam(constraint=constraints.real)
    def M_Mu(self):
        Mu = np.eye(self._n_polarities, dtype="float32")
        Mu[1, 1] = 0.3
        Mu = np.tile(Mu, (self.n_rels, 1)).reshape(
                self._n_rels, self._n_polarities, self._n_polarities
        )
        return torch.tensor(Mu)

    @PyroParam
    def M_Sigma(self):
        return torch.tensor(np.eye(self._n_polarities, dtype="float32"))

    @PyroSample
    def M(self):
        return MultivariateNormal(self.M_Mu, self.M_Sigma)

    @PyroParam
    def beta_p(self):
        return 5. * torch.tensor(
            np.ones((1, self.n_polarities), dtype="float32")
        )

    @PyroParam
    def beta_q(self):
        return 5. * torch.tensor(
            np.ones((1, self.n_polarities), dtype="float32")
        )

    @PyroSample
    def beta(self):
        return Beta(self.beta_p, self.beta_q).expand(
                            [self.n_rels, self.n_polarities]
                            ).to_event(2)

    @PyroParam
    def beta_p(self):
        return 5. * torch.tensor(
            np.ones((1, self.n_polarities), dtype="float32")
        )

    @PyroParam
    def beta_q(self):
        return 5. * torch.tensor(
            np.ones((1, self.n_polarities), dtype="float32")
        )

    @PyroSample
    def beta(self):
        raise NotImplementedError

        # create separate prior for every relation
        self.beta = nn.Parameter(torch.ones(n_rels, n_polarities))
        self.z_epsilon = nn.Parameter(torch.tensor(1e-2))
        self.scale_factor = nn.Parameter(torch.tensor(21.))
        self._min_sum = torch.tensor(1e-10)

    def forward(self, var_sfx, prnt_probs, child_probs, rels):
        # The vector `nz_chld_indices` will contain indices of the child_probs
        # which are not zero.
        nz_chld_indices = child_probs.sum(dim=-1).nonzero().squeeze(-1)
        if nz_chld_indices.nelement() == 0:
            return None, None, None, None
        # Only leave `parent`, `child`, and `rels` elements for which
        # `child_probs` are not zero.
        child_probs = child_probs[nz_chld_indices].unsqueeze_(-1)
        rels = rels[nz_chld_indices]
        # Batch-multiply the remaining child scores with relation matrices (M).
        child_probs = torch.bmm(self.M[rels], child_probs).squeeze_(-1)
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
            beta = self.beta[rels]
            alpha = (1. - beta) * prnt_probs + beta * child_probs
            scale = self.scale(prnt_probs, child_probs).unsqueeze_(-1)
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

    def scale(self, prnt_probs, child_probs):
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
        scale = self.scale_factor * cos / entropy
        return scale
