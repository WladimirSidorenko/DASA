#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet based on the
    linear combination of EDU scores.

Attributes:
  WangAnalyzer (class): class for predicting polarity of a tweet based on
    the polarity of the root EDU(s)

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from math import ceil
from torch.utils.data import DataLoader
from typing import List, Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from .constants import BUFFER_FACTOR, IDX2CLS, N_POLARITIES
from .dataset import Dataset
from .dl import DATALOADER_KWARGS, DLBaseAnalyzer


##################################################################
# Variables and Constants
NONE_IDX = 0
IRD_IDX = 1                     # index for all irrelevant discourse relations
AUX_RELS = 2


##################################################################
# Class
class Wang(nn.Module):
    """Neural network model for linear combination of EDU scores.

    """
    def __init__(self, nrels):
        """Class constructor.

        Args:
          nrels (int): total number of relations

        """
        super(Wang, self).__init__()
        d = np.ones((nrels, 1), dtype=np.float32)
        d[0] = 0
        self.d = nn.Parameter(torch.tensor(d))
        self.b = nn.Parameter(torch.tensor(0.5))

    def forward(self, rel_indices, x):
        rel_coeffs = self.d[rel_indices]
        ret = torch.sum(rel_coeffs * x, dim=1) + self.b
        return F.softmax(ret, dim=-1)


class WangAnalyzer(DLBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    """
    _name = "Wang"

    @staticmethod
    def _replace_rel(data, from_idx, to_idx):
        """Replace given relation index with another index in the dataset.

        Args:
          data (torch.utils.data.DataLoader): data container to modify
          from_idx (int): relation index that should be replaces
          to_idx (int): replacement

        Returns:
          torch.Tensor: tensor for storing indices of modified elements

        Note:
          modifies `data` in place

        """
        rels = data.dataset.tensors[0]
        mask = rels.eq(from_idx)
        new_val = mask.type_as(rels) * to_idx
        rels.data = torch.where(mask, new_val, rels)
        return mask

    @staticmethod
    def _restore_rel(mask, data, idx):
        """Restor given relation index in the dataset.

        Args:
          mask (torch.Tensor): indices of the elements that should be restored
          data (torch.utils.data.DataLoader): data container to modify
          idx (int): relation index that should be restored

        Returns:
          void

        Note:
          modifies `data` in place

        """
        rels = data.dataset.tensors[0]
        old_val = mask.type_as(rels) * idx
        rels.data = torch.where(mask, old_val, rels)

    def __init__(self, relation_scheme: str, sentiment_scores: str,
                 n_classes: int):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super().__init__(sentiment_scores, n_classes)
        self.relation_scheme = relation_scheme
        # maximum number of leaves in an RST tree
        self._max_nodes = -1
        self._rel2idx = {None: NONE_IDX, "IRD": IRD_IDX}
        self._wbench_node_scores = None
        self._wbench_rels = None

    def _train(self, train_set, dev_set=None):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list or None):
            training set
          dev_set (list or None):
            development set

        Returns:
          float: best macro-averaged F1 observed on the dev set

        """
        # determine which kind of relations to keep
        self.n_rels = n_rels = len(self._rel2idx)
        #  if we only have auxiliary relations, the just train on them
        self._logger.debug("Training model on the complete relation set.")
        best_f1 = self._train_model(train_set, dev_set, self.n_rels)
        self._logger.debug("Overall F1 score: %f", best_f1)
        best_model = self._model
        if self.n_rels <= AUX_RELS:
            return best_f1
        rels = list(self._rel2idx.keys())
        while n_rels > AUX_RELS:
            best_i = -1
            best_iteration_f1 = 0
            best_iteration_model = None
            for i, rel_i in enumerate(rels):
                rel_idx = self._rel2idx[rel_i]
                if rel_idx < AUX_RELS:
                    continue
                self._logger.debug(
                    "Temporarily assigning relation %s to the irrelevant set.",
                    rel_i
                )
                # assign relation `rel_i` to the set of irrelevant discourse
                # relations
                changed_indices_train = self._replace_rel(
                    train_set, rel_idx, IRD_IDX
                )
                changed_indices_dev = self._replace_rel(
                     dev_set, rel_idx, IRD_IDX
                )
                # we keep the initial number of relations in each new model in
                # order not to relabel the complete training set
                f1 = self._train_model(train_set, dev_set, self.n_rels)
                self._logger.debug("F1 score: %f", f1)
                if f1 > best_iteration_f1:
                    self._logger.debug(
                        "Best iteration F1 score improved from %f to %f.",
                        best_iteration_f1, f1)
                    best_i = i
                    best_iteration_f1 = f1
                    best_iteration_model = self._model
                self._logger.debug(
                    "Restoring relation %s to its original value.", rel_i)
                self._restore_rel(changed_indices_train, train_set, rel_idx)
                self._restore_rel(changed_indices_dev, dev_set, rel_idx)
            if best_iteration_f1 > best_f1:
                rel2remove = rels[best_i]
                self._logger.debug(
                    "Best overall F1 score improved from %f to %f "
                    "(permanently assigning relation %s to the irrelevant"
                    " set)", best_f1, best_iteration_f1, rel2remove
                )
                best_f1 = best_iteration_f1
                best_model = best_iteration_model
                # permanently replace the `best_i` relation in the datasets
                rel_idx = self._rel2idx[rel_i]
                self._rel2idx[rel_i] = IRD_IDX
                self._replace_rel(train_set, rel_idx, IRD_IDX)
                self._replace_rel(dev_set, rel_idx, IRD_IDX)
                n_rels -= 1
            else:
                self._logger.debug(
                    "Best overall F1 (%f) score did not improve.", best_f1
                )
                break
        self._model = best_model
        self._logger.debug("Best observed F1 score: %f", best_f1)
        return best_f1

    def _train_model(self, train_set, dev_set, n_rels):
        self._model = Wang(n_rels)
        return super()._train(train_set, dev_set)

    def predict_instance(self, instance: dict,
                         relation_scheme: Optional[str] = None,
                         sentiment_scores: Optional[str] = None) -> str:
        leaves = [n
                  for n in self.build_rst(instance, relation_scheme,
                                          sentiment_scores)
                  if n.is_leaf]
        if len(leaves) > self._max_nodes:
            self._max_nodes = len(leaves)
            self._wbench_node_scores.resize((1, self._max_nodes, N_POLARITIES))
            self._wbench_rels.resize((1, self._max_nodes))
        self._wbench_node_scores *= 0
        self._wbench_rels *= 0
        edus = instance["edus"]
        for i, leaf in enumerate(leaves):
            self._wbench_node_scores[0, i, :] = self._get_scores(edus[leaf.id])
            rel = (leaf.rel2par, leaf.ns)
            self._wbench_rels[0, i] = self._rel2idx.get(rel, IRD_IDX)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench_rels),
                torch.from_numpy(self._wbench_node_scores),
            )
        _, cls_idx = torch.max(out, 1)
        cls = IDX2CLS[cls_idx.item()]
        return cls

    def debug(self, instance,
              relation_scheme: Optional[str] = None,
              sentiment_scores: Optional[str] = None) -> str:
        self._logger.debug("instance: %r", instance)
        leaves = [n
                  for n in self.build_rst(instance, relation_scheme,
                                          sentiment_scores)
                  if n.is_leaf]
        self._logger.debug("leaves: %r", leaves)
        if len(leaves) > self._max_nodes:
            self._max_nodes = len(leaves)
            self._wbench_node_scores.resize((1, self._max_nodes, N_POLARITIES))
            self._wbench_rels.resize((1, self._max_nodes))
        self._wbench_node_scores *= 0
        self._wbench_rels *= 0
        edus = instance["edus"]
        for i, leaf in enumerate(leaves):
            self._wbench_node_scores[0, i, :] = self._get_scores(edus[leaf.id])
            rel = (leaf.rel2par, leaf.ns)
            self._wbench_rels[0, i] = self._rel2idx.get(rel, IRD_IDX)
        self._logger.debug("rels: %r", self._wbench_rels)
        self._logger.debug("node_scores: %r", self._wbench_node_scores)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench_rels),
                torch.from_numpy(self._wbench_node_scores),
            )
        self._logger.debug("out: %r", out)
        _, cls_idx = torch.max(out, 1)
        cls = IDX2CLS[cls_idx.item()]
        self._logger.debug("cls_idx: %d (%s)", cls_idx, cls)
        return cls

    def _digitize_data(self, X: List[dict], Y: np.array, train_mode=False):
        rst_leaves = [
            [
                (
                    self._get_scores(instance["edus"][node.id]),
                    (node.rel2par, node.ns)
                )
                for node in self.build_rst(instance)
                if node.is_leaf
            ]
            for instance in X
        ]
        if self._max_nodes < 0:
            self._max_nodes = ceil(
                BUFFER_FACTOR * max([len(nodes) for nodes in rst_leaves])
            )
            self._init_wbenches()
        n = len(X)
        digitized_rels = np.zeros((n, self._max_nodes), dtype="long")
        digitized_input = np.zeros((n, self._max_nodes, N_POLARITIES),
                                   dtype="float32")
        for i, (leaves, instance) in enumerate(zip(rst_leaves, X)):
            for j, (pol_scores, rel) in enumerate(leaves):
                if rel not in self._rel2idx:
                    if train_mode:
                        self._rel2idx[rel] = len(self._rel2idx)
                    else:
                        self._logger.warn(
                            "Unknown discourse relation discovered"
                            " at test time: %r (using IRD)", rel
                        )
                digitized_rels[i, j] = self._rel2idx.get(rel, IRD_IDX)
                digitized_input[i, j, :] = pol_scores
        dataset = Dataset(digitized_rels, digitized_input, Y)
        return DataLoader(dataset, **DATALOADER_KWARGS)

    def _init_wbenches(self):
        """Initialize workbenches that will be used at prediction time.

        """
        self._wbench_node_scores = np.zeros(
            (1, self._max_nodes, N_POLARITIES), dtype="float32"
        )
        self._wbench_rels = np.zeros((1, self._max_nodes), dtype="long")
