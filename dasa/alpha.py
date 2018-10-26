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
            np.ones((n_rels, n_polarities, n_polarities), dtype="float32"))
        self.p = 15. * torch.tensor(np.ones(n_rels, dtype="float32"))
        self.q = 15. * torch.tensor(np.ones(n_rels, dtype="float32"))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, var_sfx, prnt_probs, child_probs, rels):
        # This vector will consist of either 0's or 1's, depending on whether
        # the child is a real or auxiliary node.  In the latter case, we don't
        # want the child to affect the probability of the parent vector.
        child_sum = child_probs.sum(dim=-1, keepdim=True)
        if child_sum.sum() == 0:
            return None
        # sample relation matrices `M_ij` for each unique relation `i-j` (these
        # matrices will be responsible for swapping the context)
        M = pyro.sample("M_{}".format(var_sfx), dist.Normal(
            self.rel_mu[rels], self.rel_sigma[rels]).independent(2))
        # print("M: ", repr(M))
        # batch-multiply sampled relation matrices with the correponding child
        # scores
        child_probs.unsqueeze_(-1)
        # print("child_probs [f]: ", repr(child_probs), child_probs.shape)
        child_probs = self.softmax(torch.bmm(M, child_probs))
        child_probs.squeeze_()
        # print("child_probs [f]: ", repr(child_probs))
        # print("self.p[rels] [f]: ", repr(self.p[rels]), self.p[rels].shape)
        # print("self.q[rels] [f]: ", repr(self.q[rels]), self.q[rels].shape)
        beta = pyro.sample(
            "beta_{}".format(var_sfx),
            dist.Beta(self.p[rels], self.q[rels]).independent(1)
        ).unsqueeze(-1)
        print("beta [f]: ", repr(beta), beta.shape)
        scaled_beta = beta * child_sum
        # print("child_sum [f]: ", repr(child_sum), child_sum.shape)
        # print("1. - beta [f]: ", repr(1. - beta), (1. - beta).shape)
        # print("prnt_probs [f]: ", repr(prnt_probs), prnt_probs.shape)
        # print("child_probs [f]: ", repr(child_probs), child_probs.shape)
        # take a convex combination of the parent and child scores as new
        # values for alpha (since some of the vect)
        prnt_sum = prnt_probs.sum(dim=-1, keepdim=True)
        alpha = (1. - scaled_beta) * prnt_sum * prnt_probs \
            + scaled_beta * child_probs
        scale = self.scale(prnt_probs, child_probs).unsqueeze_(-1)
        # print("alpha mu [f]:", repr(alpha), alpha.shape)
        # print("scale [f]:", repr(scale), scale.shape)
        alpha *= scale
        # print("alpha scaled [f]:", repr(alpha))
        return alpha

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
    pass
