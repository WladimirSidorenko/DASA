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

from six import iteritems, string_types
from torch.distributions import constraints
import numpy as np
import pyro.distributions as dist
import pyro
import torch.nn as nn
import torch

from .constants import CONTRASTIVE_RELS


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
# Classes
class AlphaBase(nn.Module):
    def __init__(self):
        super(AlphaBase, self).__init__()
        self.cos_epsilon = torch.tensor(1e-2)
        self.z_epsilon = torch.tensor(1e-2)
        self.scale_factor = torch.tensor(42., dtype=torch.float32)


class AlphaModel(AlphaBase):
    def __init__(self, rel2idx, n_polarities=3):
        super(AlphaModel, self).__init__()
        # create separate prior for every relation
        MU = np.eye(n_polarities, dtype="float32")
        MU[1, :] = [-0.25, 0.5, -0.25]
        n_rels = len(rel2idx)
        MU = np.tile(MU, (n_rels, 1)).reshape(
            n_rels, n_polarities, n_polarities
        )
        for rel, rel_idx in iteritems(rel2idx):
            # swap axes for contrastive relations
            if check_rel(rel, CONTRASTIVE_RELS):
                mu_i = MU[rel_idx]
                mu_i[[0, 2]] = mu_i[[2, 0]]
        self.rel_mu = torch.tensor(MU)
        self.rel_sigma = torch.tensor(
            np.ones((n_rels, n_polarities, n_polarities),
                    dtype="float32")
        )
        self.beta_p = 15. * torch.tensor(np.ones(n_rels, dtype="float32"))
        self.beta_q = 15. * torch.tensor(np.ones(n_rels, dtype="float32"))
        self.softmax = nn.Softmax(dim=-1)

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
        # Sample relation matrices `M_ij` for each unique relation `i-j` (these
        # matrices will be responsible for swapping the context).
        M = pyro.sample(
            "M_{}".format(var_sfx),
            dist.Normal(self.rel_mu[rels], self.rel_sigma[rels]).independent(2)
        )
        # Batch-multiply sampled relation matrices with the correponding child
        # scores.
        child_probs = self.softmax(torch.bmm(M, child_probs).squeeze_(-1))
        # We will only take a convex combination of the means if the parent
        # probs are also non-zero, otherwise we will assign the child probs to
        # the parents as is.
        prnt_probs = prnt_probs[nz_chld_indices]
        prnt_probs_sum = prnt_probs.sum(dim=-1)
        z_prnt_indices = (prnt_probs_sum == 0.).nonzero().squeeze_(-1)
        # indices of instances whose child scores are non-zero, but parent
        # scores are zero
        copy_indices = nz_chld_indices[z_prnt_indices]
        child_probs2copy = child_probs[z_prnt_indices]
        nz_prnt_indices = prnt_probs_sum.nonzero().squeeze(-1)
        alpha_indices = nz_chld_indices[nz_prnt_indices]
        if alpha_indices.numel() == 0:
            alpha = None
        else:
            child_probs = child_probs[nz_prnt_indices]
            rels = rels[nz_prnt_indices]
            prnt_probs = prnt_probs[nz_prnt_indices]
            beta = pyro.sample(
                "beta_{}".format(var_sfx),
                dist.Beta(self.beta_p[rels], self.beta_q[rels]).independent(1)
            ).unsqueeze(-1)
            # take a convex combination of the parent and child scores as new
            # values for alpha (since some of the vect)
            alpha = (1. - beta) * prnt_probs + beta * child_probs
            scale = self.scale(prnt_probs, child_probs).unsqueeze_(-1)
            alpha *= scale
        return copy_indices, child_probs2copy, alpha_indices, alpha

    def scale(self, prnt_probs, child_probs):
        min_score = self.z_epsilon * torch.ones(prnt_probs.shape)
        z = torch.max(min_score, prnt_probs + child_probs)
        # print("z", repr(z), z.shape)
        z_norm = 1./torch.sum(z, dim=-1)
        z_norm.unsqueeze_(-1)
        # print("z_norm", repr(z_norm), z_norm.shape)
        z *= z_norm
        # print("normalized z:", repr(z))
        entropy = -torch.sum(z * torch.log(z), dim=-1)
        # print("entropy:", repr(entropy))
        cos = torch.sum(prnt_probs * child_probs, dim=-1)
        # print("unnormalized cosine:", repr(cos))
        norm = torch.norm(prnt_probs, dim=-1) * torch.norm(child_probs, dim=-1)
        # replace 0's with 1's to prevent division by 0, since cosine in
        # this case will be 0 anyway
        norm = torch.where(norm == 0, torch.ones(norm.shape), norm)
        # print("norm:", repr(norm))
        cos /= norm
        cos = torch.max(self.cos_epsilon * torch.ones(cos.shape), cos)
        # print("normalized cosine:", repr(cos))
        scale = self.scale_factor * cos / entropy
        # print("scale", repr(scale))
        return scale


class AlphaGuide(AlphaModel):
    def __init__(self, rel2idx, n_polarities=3):
        super(AlphaGuide, self).__init__(rel2idx, n_polarities)
        self.rel_mu = pyro.param("rel_mu", self.rel_mu)
        self.rel_sigma = pyro.param("rel_sigma", self.rel_sigma,
                                    constraint=constraints.positive)
        self.beta_p = pyro.param("beta_p", self.beta_p,
                                 constraint=constraints.positive)
        self.beta_q = pyro.param("beta_q", self.beta_q,
                                 constraint=constraints.positive)
