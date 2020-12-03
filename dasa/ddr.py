#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet based on
discourse-depth reweighting.

Attributes:
  DDRAnalyzer (class): class for predicting polarity of a tweet based on the
    polarity of the root EDU(s)

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from torch import eye
from torch.utils.data import DataLoader
from typing import List, Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from .constants import CLS2IDX, IDX2CLS
from .dataset import Dataset
from .dl import DLBaseAnalyzer


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

    def forward(self, x):
        ret = super(DDR, self).forward(x)
        return F.softmax(ret, dim=-1)


class DDRAnalyzer(DLBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    _name = "DDR"

    def __init__(self, relation_scheme: str, sentiment_scores: str,
                 n_classes: int):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super().__init__(sentiment_scores, n_classes)
        self.relation_scheme = relation_scheme
        self._model = DDR()

    def predict_instance(self, instance: dict,
                         relation_scheme: Optional[str] = None,
                         sentiment_scores: Optional[str] = None) -> str:
        self._wbench *= 0
        self._compute_scores(self._wbench[0, :], instance,
                             relation_scheme, sentiment_scores)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench)
            )
            _, cls_idx = torch.max(out, 1)
        return IDX2CLS[cls_idx.item()]

    def debug(self, instance: dict,
              relation_scheme: Optional[str] = None,
              sentiment_scores: Optional[str] = None) -> str:
        self._logger.info("instance: %r", instance)
        self._wbench *= 0
        self._logger.info("wbench: %r", self._wbench)
        self._compute_scores(self._wbench[0, :], instance,
                             relation_scheme, sentiment_scores)
        self._logger.info("* wbench: %r", self._wbench)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench).double()
            )
        self._logger.info("out: %r", out)
        _, cls_idx = torch.max(out, 1)
        self._logger.info("cls_idx: %r (%s)",
                          cls_idx, IDX2CLS[cls_idx.item()])
        return IDX2CLS[cls_idx.item()]

    def _digitize_data(self, X: List[dict], Y: np.array,
                       train_mode: bool = False) -> DataLoader:
        n = len(X)
        m = len(CLS2IDX)
        digitized_input = np.zeros((n, m), dtype="float32")
        for i, instance in enumerate(X):
            self._compute_scores(digitized_input[i, :], instance)
        dataset = Dataset(digitized_input, Y)
        return DataLoader(dataset, **self.DATALOADER_KWARGS)

    def _compute_scores(self, scores: np.array, instance: dict,
                        relation_scheme: Optional[str] = None,
                        sentiment_scores: Optional[str] = None) -> np.array:
        rst_tree = self.build_rst(instance, relation_scheme, sentiment_scores)
        dep_tree = rst_tree.to_deps()
        for i, nodes_i in enumerate(dep_tree.bfs(), -1):
            if i < 0:
                continue
            lambda_i = max(0.5, 1. - float(i)/6.)
            scores += lambda_i * np.sum([n.polarity_scores
                                         for n in nodes_i], axis=0)
        return scores
