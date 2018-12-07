#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing implementation of `spasemax` operation.

This code was graefully copied from
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py

See

> [SparseMAP: Differentiable Sparse Structured Inference](https://arxiv.org/abs/1802.04223).
> Vlad Niculae, Andre F.T. Martins, Mathieu Blondel, Claire Cardie.
> In: Proc. of ICML, 2018.

Attributes:

"""

##################################################################
# Imports
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch.nn as nn
import torch


##################################################################
# Variables and Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################################################
# Class
class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
          dim (int, optional): The dimension over which to apply the sparsemax
            function.

        """
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.

        Args:
          input (torch.Tensor): Input tensor. First dimension should be the
            batch size

        Returns:
          torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim,
                                  keepdim=True)[0].expand_as(input)

        # Sort input in descending order.  (NOTE: Can be replaced with linear
        # time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1.,
                             device=device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / \
            torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input
