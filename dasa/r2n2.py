#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet based on the
    polarity of the root EDU(s).

Attributes:
  R2N2Analyzer (class): class for predicting polarity of a tweet using
    rhetorical recursive neural network

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from builtins import range
from math import ceil
from six import iteritems
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from .constants import BUFFER_FACTOR, CLS2IDX, IDX2CLS, N_POLARITIES
from .dataset import Dataset
from .dl import DATALOADER_KWARGS, DLBaseAnalyzer
from .rst import Tree as RSTTree


##################################################################
# Variables and Constants


##################################################################
# Class
class R2N2(nn.Module):
    """Neural network for recursively predicting sentiment scores using RST tree.

    """

    def __init__(self, rels):
        """Class constructor.

        Args:
          rels (set): set of relation tuples

        """
        super(R2N2, self).__init__()
        # initialize mapping from relations to indices
        self._rel2idx = {rel_i: i for i, rel_i in enumerate(rels, 1)}
        # initialize relation matrices
        n_rels = len(self._rel2idx)
        K = torch.eye(N_POLARITIES).repeat(n_rels + 1, 1).reshape(
                [-1, N_POLARITIES, N_POLARITIES])
        K[0] *= 0
        self.K = nn.Parameter(K)
        # initialize scalar parameter
        self.gamma = nn.Parameter(torch.tensor(0.5))

    @property
    def rel2idx(self):
        return self._rel2idx

    def forward(self, node_scores, children, rels, msg_scores):
        batch_indices = torch.arange(node_scores.shape[0], dtype=torch.long)
        batch_indices.unsqueeze_(-1)
        batch_indices = batch_indices.expand(
            (-1, node_scores.shape[2])
        ).contiguous().view(-1)
        for i in range(1, node_scores.shape[1]):
            # obtain scores of child nodes
            child_indices = children[:, i].contiguous().view(-1)
            child_scores = node_scores[batch_indices, child_indices]
            child_scores.unsqueeze_(1)
            # obtain relation matrices
            rel_indices = rels[:, i].contiguous().view(-1)
            rel_scores = self.K[rel_indices]
            # batch multiplication of accumulated child scores and
            # corresponding relation matrices
            update_scores = torch.bmm(child_scores, rel_scores).view(
                (node_scores.shape[0], -1, node_scores.shape[2])
            )
            update_scores = F.tanh(torch.sum(update_scores, dim=1))
            # add new scores to the existing ones
            node_scores[:, i] += update_scores
        return F.softmax(self.gamma * msg_scores + node_scores[:, -1], dim=-1)


class R2N2Analyzer(DLBaseAnalyzer):
    """Discourse-aware sentiment analysis using rhetorical recursive neural models.

    Attributes:

    """

    def __init__(self, relation_scheme, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(R2N2Analyzer, self).__init__(*args, **kwargs)
        self._name = "R2N2"
        self._relation_scheme = relation_scheme
        self._relations = {}
        self._max_nodes = -1    # maximum depth of an RST tree
        self._max_width = -1    # maximum number of children in an RST node
        self._wbench_node_scores = None
        self._wbench_children = None
        self._wbench_rels = None
        self._wbench_msg_scores = None
        self.root_offset = 1

    def predict(self, instance):
        tree = self.span2nuc(
            RSTTree(
                instance, instance["rst_trees"][self._relation_scheme])
        )
        self.tree2mtx(self._wbench_node_scores[0, :],
                      self._wbench_children[0, :],
                      self._wbench_rels[0, :],
                      tree, instance)
        self._wbench_msg_scores[0, :] = instance["polarity_scores"]
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench_node_scores),
                torch.from_numpy(self._wbench_children),
                torch.from_numpy(self._wbench_rels),
                torch.from_numpy(self._wbench_msg_scores),
            )
        _, cls_idx = torch.max(out, 1)
        return IDX2CLS[cls_idx.item()]

    def debug(self, instance):
        tree = self.span2nuc(
            RSTTree(
                instance, instance["rst_trees"][self._relation_scheme])
        )
        self._logger.info("RST tree: %r", tree)
        self.tree2mtx(self._wbench_node_scores[0, :],
                      self._wbench_children[0, :],
                      self._wbench_rels[0, :],
                      tree, instance)
        self._logger.info("node_scores: %r", self._wbench_node_scores)
        self._logger.info("children: %r", self._wbench_children)
        self._logger.info("rels: %r", self._wbench_rels)
        self._wbench_msg_scores[0, :] = instance["polarity_scores"]
        self._logger.info("msg_scores: %r", self._wbench_msg_scores)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(self._wbench_node_scores),
                torch.from_numpy(self._wbench_children),
                torch.from_numpy(self._wbench_rels),
                torch.from_numpy(self._wbench_msg_scores),
            )
        self._logger.info("out: %r", out)
        max_scores, cls_idx = torch.max(out, 1)
        self._logger.info("max_scores: %r", max_scores)
        self._logger.info("cls_idx: %r (%s)", cls_idx, IDX2CLS[cls_idx.item()])
        return IDX2CLS[cls_idx.item()]

    def _digitize_data(self, data, train_mode=False):
        forrest = [
            self.span2nuc(
                RSTTree(
                    instance, instance["rst_trees"][self._relation_scheme])
            )
            for instance in data]
        if self._max_nodes < 0:
            self._max_width = ceil(max([t.width for t in forrest])
                                   * BUFFER_FACTOR)
            self._max_nodes = ceil(max([len(t) for t in forrest])
                                   * BUFFER_FACTOR + 1)
            self._init_model(forrest)
            self._init_wbenches()
        n = len(data)
        node_scores = np.zeros((n, self._max_nodes, N_POLARITIES),
                               dtype="float32")
        msg_scores = np.zeros((n, N_POLARITIES), dtype="float32")
        children = np.zeros((n, self._max_nodes, self._max_width),
                            dtype="long")
        rels = np.zeros((n, self._max_nodes, self._max_width),
                        dtype="long")
        labels = np.zeros(n, dtype="long")
        for i, (tree_i, instance_i) in enumerate(zip(forrest, data)):
            self.tree2mtx(node_scores[i, :], children[i, :],
                          rels[i, :], tree_i, instance_i)
            msg_scores[i] = instance_i["polarity_scores"]
            labels[i] = CLS2IDX[instance_i["label"]]
        dataset = Dataset(node_scores, children, rels, msg_scores, labels)
        return DataLoader(dataset, **DATALOADER_KWARGS)

    def _init_model(self, forrest):
        """Initialize the model that will be used for prediction.

        Args:
          forrest (list[RSTTree]): list of all training RST trees

        """
        rels = self.get_rels(forrest)
        self._model = R2N2(rels)

    def _init_wbenches(self):
        """Initialize workbenches that will be used at prediction time.

        """
        self._wbench_node_scores = np.zeros(
            (1, self._max_nodes, N_POLARITIES),
            dtype="float32"
        )
        self._wbench_children = np.zeros(
            (1, self._max_nodes, self._max_width),
            dtype="long"
        )
        self._wbench_rels = np.zeros(
            (1, self._max_nodes, self._max_width),
            dtype="long"
        )
        self._wbench_msg_scores = np.zeros(
            (1, N_POLARITIES),
            dtype="float32"
        )

    def tree2mtx(self, node_scores, children, rels, tree, instance):
        """Convert an RST tree to a matrix of scores and node indices.

        Args:
          node_scores (np.array): target matrix of node scores
          children (np.array): target matrix of child indices
          rels (np.array): target matrix of relation indices
          tree (rst.Tree): source RST tree
          instance (dict): tweet instance whose tree is being converted

        Returns:
          None

        Note:
          modifies `scores` and `children` in place

        """
        assert self._model is not None, (
            "Model must be initialized before digitizing the dataset."
        )
        node_scores *= 0
        children *= 0
        rels *= 0
        nodes = [node for node in tree]
        # mapping from node id to node
        n_nodes = len(nodes)
        assert n_nodes < self._max_nodes, (
            "Number of nodes in RST tree ({:d}) exceeds"
            " the maximum allowed node number ({:d})."
        ).format(n_nodes, self._max_nodes)
        # establish mapping from node id's to array indices
        node_id2node = dict()
        node_id2mtx_idx = dict()
        max_T = self._max_nodes - self.root_offset
        for node, mtx_idx in zip(nodes, range(max_T, -1, -1)):
            n_id = node.id
            node_id2node[n_id] = node
            node_id2mtx_idx[n_id] = mtx_idx
        for node_id, mtx_idx in iteritems(node_id2mtx_idx):
            node = node_id2node[node_id]
            if node_id >= 0:
                node_scores[mtx_idx, :] = node.polarity_scores
            for i, ch in enumerate(node.children):
                children[mtx_idx, i] = node_id2mtx_idx[ch.id]
                rel_name = (ch.rel2par, ch.ns)
                if rel_name not in self._model.rel2idx:
                    self._logger.warn("Unknown relation %r", rel_name)
                rel_id = self._model.rel2idx.get(rel_name, 0)
                rels[mtx_idx, i] = rel_id
