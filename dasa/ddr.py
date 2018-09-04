#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet based on the
    polarity of the root EDU(s).

Attributes:
  DASRootAnalyzer (class): class for predicting polarity of a tweet based on
    the polarity of the root EDU(s)

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from torch import eye
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch

from .constants import CLS2IDX, DFLT_DTYPE, IDX2CLS
from .dataset import Dataset
from .dl import DATALOADER_KWARGS, DLBaseAnalyzer
from .rst import Tree as RSTTree


##################################################################
# Variables and Constants


##################################################################
# Class
class DDR(nn.Linear):
    """Neural network model for doing DDR prediction.

    """
    def __init__(self):
        n = len(CLS2IDX)
        super(DDR, self).__init__(n, n)
        self.weight.data.copy_(eye(n))
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ret = super(DDR, self).forward(x)
        return self._softmax(ret)


class DDRAnalyzer(DLBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    def __init__(self, relation_scheme, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(DDRAnalyzer, self).__init__(*args, **kwargs)
        self._name = "DDR"
        self._model = DDR()
        self._relation_scheme = relation_scheme

    def predict(self, instance):
        self._wbench *= 0
        self._compute_scores(self._wbench[0, :], instance)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench).to(DFLT_DTYPE)
            )
        _, cls_idx = torch.max(out, 1)
        return IDX2CLS[cls_idx.item()]

    def debug(self, instance):
        self._logger.info("instance: %r", instance)
        self._wbench *= 0
        self._logger.info("wbench: %r", self._wbench)
        self._compute_scores(self._wbench[0, :], instance)
        self._logger.info("* wbench: %r", self._wbench)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench).to(DFLT_DTYPE)
            )
        self._logger.info("out: %r", out)
        _, cls_idx = torch.max(out, 1)
        self._logger.info("cls_idx: %r (%s)",
                          cls_idx, IDX2CLS[cls_idx.item()])
        return IDX2CLS[cls_idx.item()]

    def _digitize_data(self, data):
        n = len(data)
        m = len(CLS2IDX)
        digitized_input = np.zeros((n, m), dtype="float32")
        digitized_labels = np.zeros(n, dtype="long")
        for i, instance in enumerate(data):
            self._compute_scores(digitized_input[i, :], instance)
            digitized_labels[i] = CLS2IDX[instance["label"]]
        dataset = Dataset(digitized_input, digitized_labels)
        return DataLoader(dataset, **DATALOADER_KWARGS)

    def _compute_scores(self, scores, instance):
        rst_tree = RSTTree(instance,
                           instance["rst_trees"][self._relation_scheme])
        dep_tree = rst_tree.to_deps()
        for i, nodes_i in enumerate(dep_tree.bfs(), -1):
            if i < 0:
                continue
            lambda_i = max(0.5, 1. - float(i)/6.)
            scores += lambda_i * np.sum([n.polarity_scores
                                         for n in nodes_i], axis=0)
        return scores
