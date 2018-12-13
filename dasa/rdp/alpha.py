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

import torch.nn as nn
import torch

from .sparsemax import Sparsemax


##################################################################
# Classes
class AlphaModel(nn.Module):
    def __init__(self, n_rels, n_polarities=3):
        super(AlphaModel, self).__init__()
        # create separate prior for every relation
        self.M = nn.Parameter(torch.ones(n_rels, n_polarities, n_polarities))
        self.beta = nn.Parameter(torch.ones(n_rels, n_polarities))
        self.z_epsilon = nn.Parameter(torch.tensor(1e-2))
        self.scale_factor = nn.Parameter(torch.tensor(21.))
        self._min_sum = torch.tensor(1e-10)
        self._sparsemax = Sparsemax(dim=-1)
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, var_sfx, prnt_probs, child_probs, rels):
        # The vector `nz_chld_indices` will contain indices of child_probs
        # which are not all zero.
        n = child_probs.shape[1]
        nz_chld_indices = (
            (child_probs == 0).sum(1) != n).nonzero().squeeze(-1)
        if nz_chld_indices.nelement() == 0:
            return None, None, None, None
        # Only leave `parent`, `child`, and `rels` elements for which
        # `child_probs` are not zero.
        child_probs = child_probs[nz_chld_indices].unsqueeze_(-1)
        rels = rels[nz_chld_indices]
        child_probs = torch.bmm(self.M[rels], child_probs).squeeze_(-1)
        # Batch-multiply the remaining child scores with relation matrices (M).
        # Three different normalization variants:
        # Softmax
        # child_probs = self._softmax(child_probs)
        # Sparsemax
        child_probs = self._sparsemax(child_probs)
        # Custom Normalization
        # child_probs = self._normalize(child_probs)
        # We will only take a convex combination of the means if the parent
        # probs are also non-zero, otherwise we will assign the child probs to
        # the parents unchanged.
        prnt_probs = prnt_probs[nz_chld_indices]
        m = prnt_probs.shape[1]
        z_prnt_indices = ((prnt_probs == 0).sum(1) == m).nonzero().squeeze_(-1)
        # indices of instances whose child scores are non-zero, but parent
        # scores are zero
        copy_indices = nz_chld_indices[z_prnt_indices]
        if len(z_prnt_indices):
            child_probs2copy = self._sparsemax(child_probs[z_prnt_indices])
        else:
            child_probs2copy = child_probs[z_prnt_indices]
        nz_prnt_indices = ((prnt_probs == 0).sum(1) != m).nonzero().squeeze_(
            -1
        )
        alpha_indices = nz_chld_indices[nz_prnt_indices]
        if alpha_indices.nelement() == 0:
            alpha = None
        else:
            child_probs = self._sparsemax(child_probs[nz_prnt_indices])
            prnt_probs = self._sparsemax(prnt_probs[nz_prnt_indices])
            rels = rels[nz_prnt_indices]
            # take a convex combination of the parent and child scores as new
            # values for alpha and scale the resulting alpha scores with the
            # corresponding scale factor.
            beta = self.beta[rels]
            alpha = (1. - beta) * prnt_probs + beta * child_probs
            scale = self.scale(prnt_probs, child_probs).unsqueeze_(-1)
            alpha *= scale
            alpha = torch.clamp(alpha, min=1e-3)
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

    def _normalize(self, probs):
        """Normalize probabilities.

        Args:
          probs (torch.tensor): unnormalized probabilities

        """
        min_correction = torch.min(probs, dim=1, keepdim=True)[0] - 0.1
        probs -= min_correction
        probs /= torch.clamp(probs.sum(dim=1, keepdim=True),
                             min=self._min_sum)
        return probs


class AlphaGuide(AlphaModel):
    pass
