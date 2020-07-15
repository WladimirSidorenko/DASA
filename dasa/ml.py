#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing an abstract base for deep-learning--based analyzers.

Attributes:
  MLBaseAnalyzer (class): abstract base for all machine-learning--based
    analyzers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Iterable, List
import abc
import numpy as np

from .base import DASBaseAnalyzer
from .constants import CLS2IDX

##################################################################
# Variables and Constants


##################################################################
# Classes
class MLBaseAnalyzer(DASBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_rels(forrest):
        """Extract all relations present in forrest of RST trees.

        Args:
          forrest (list[rst.Tree]): list of RST trees

        """
        rels = set()
        for tree in forrest:
            nodes = [tree]
            while nodes:
                node = nodes.pop(0)
                nodes.extend(node.children)
                if node.rel2par is not None:
                    rels.add((node.rel2par, node.ns))
        return rels

    @staticmethod
    def span2nuc(tree):
        """Translate all span relations to immediate nucleus rels.

        Args:
          forrest (list[rst.Tree]): list of RST trees

        """
        nodes = [tree]
        while nodes:
            node = nodes.pop(0)
            if node.rel2par == "span":
                siblings = node.parent.children
                for i, sib_i in enumerate(siblings):
                    if sib_i.id != node.id:
                        node.rel2par = sib_i.rel2par
                        break
            nodes.extend(node.children)
        return tree

    def train(self, train_set, dev_set=None):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list):
            training set
          dev_set (list or None):
            development set

        Returns:
          float: best macro-averaged F1 observed on the dev set

        """
        self._logger.debug("Preparing data...")
        train_set, dev_set = self._prepare_data(train_set, dev_set)
        self._logger.debug("Data prepared...")
        return self._train(train_set, dev_set)

    @abc.abstractmethod
    def _train(self, train_set, dev_set):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list): training set
          dev_set (list): development set

        Returns:
          float: best macro-averaged F1 observed on the dev set

        """
        raise NotImplementedError

    def _prepare_data(self, X_train, X_dev):
        """Provide train/test split and digitize data.

        """
        Y_train = self._digitize_labels(X_train)
        if not X_dev:
            X_train, X_dev, Y_train, Y_dev = train_test_split(
                X_train, Y_train, test_size=0.15, stratify=Y_train
            )
        else:
            Y_dev = self._digitize_labels(X_dev)
        # convert tweets to word indices
        train_set = self._digitize_data(X_train, Y_train, train_mode=True)
        dev_set = self._digitize_data(X_dev, Y_dev, train_mode=False)
        return (train_set, dev_set)

    @abc.abstractmethod
    def _digitize_data(self, X: List[dict], Y: np.array,
                       train_mode: bool = False) -> DataLoader:
        raise NotImplementedError

    def _lbls2ints(self, labels: Iterable[str]) -> np.array:
        return np.array(
            [CLS2IDX[lbl_i] for lbl_i in labels],
            dtype="long"
        )

    def _digitize_labels(self, X: List[dict]) -> np.array:
        labels = self._extract_labels(X)
        return self._lbls2ints(labels)
