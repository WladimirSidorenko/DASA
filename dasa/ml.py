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

from builtins import range
from copy import deepcopy
import abc
import numpy as np

from .base import DASBaseAnalyzer

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
                assert len(siblings) == 2, \
                    "Multiple siblings found for a span node."
                idx2delete = -100
                nodes2add = []
                for i, sib_i in enumerate(siblings):
                    if sib_i.id == node.id:
                        idx2delete = i
                    else:
                        node = deepcopy(node)
                        node.rel2par = sib_i.rel2par
                        nodes2add.append(node)
                siblings.pop(idx2delete)
                siblings.extend(nodes2add)
            nodes.extend(node.children)
        return tree

    def train(self, train_set, dev_set=None, grid_search=True, balance=False):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list):
            training set
          dev_set (list or None):
            development set
          grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          balance (bool): balance dataset to get equal number of instances
            for all classes (via downsampling)

        Returns:
          float: best macro-averaged F1 observed on the dev set

        """
        self._logger.debug("Preparing data...")
        train_set, dev_set = self._prepare_data(train_set, dev_set)
        self._logger.debug("Data prepared...")
        return self._train(train_set, dev_set, grid_search, balance)

    @abc.abstractmethod
    def _train(self, train_set, dev_set, grid_search=True, balance=False):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list): training set
          dev_set (list): development set
          grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          balance (bool): balance dataset to get equal number of instances
            for all classes (via downsampling)


        Returns:
          float: best macro-averaged F1 observed on the dev set

        """
        raise NotImplementedError

    def _prepare_data(self, train_set, dev_set):
        """Provide train/test split and digitize the data.

        """
        if not dev_set and len(train_set) > 1:
            n = len(train_set)
            n_dev = max(1, n // 15)
            idcs = list(range(n))
            np.random.shuffle(idcs)

            def get_split(data, idcs):
                return [data[i] for i in idcs]

            dev_set = get_split(train_set, idcs[:n_dev])
            train_set = get_split(train_set, idcs[n_dev:])

        # convert tweets to word indices
        train_set = self._digitize_data(train_set, train_mode=True)
        dev_set = self._digitize_data(dev_set, train_mode=False)
        return (train_set, dev_set)

    @abc.abstractmethod
    def _digitize_data(self, data, train_mode=False):
        raise NotImplementedError
