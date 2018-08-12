#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing an abstract base for all discourse-aware analyzers.

Attributes:
  BaseAnalyzer (class): abstract base for all discourse-aware sentiment
    analyzers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

import numpy as np

from .base import DASBaseAnalyzer
from .constants import IDX2CLS
from .rst import Tree as RSTTree


##################################################################
# Variables and Constants


##################################################################
# Classes
class DASRootAnalyzer(DASBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    def __init__(self, relation_scheme, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(DASRootAnalyzer, self).__init__(*args, **kwargs)
        self._relation_scheme = relation_scheme
        self._name = "RootEDU"

    def train(self, *args, **kwargs):
        """Train specified model(s) on the provided data.

        Args:
          a_train_data (list or None):
            training set
          a_dev_data (list or None):
            development set
          a_path (str):
            path for storing the model
          a_grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          a_balance (bool): balance dataset to get equal number of instances
            for all classes (via downsampling)

        Returns:
          void:

        """
        # this analyzer does not require training
        pass

    def predict(self, instance):
        """Predict label of a single input instance.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        tree = RSTTree(instance, instance["rst_trees"][self._relation_scheme])
        roots = tree.root_edus
        scores = np.sum(r.polarity_scores for r in roots)
        cls_idx = np.argmax(scores)
        return IDX2CLS[cls_idx]

    def debug(self, instance):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        tree = RSTTree(instance, instance["rst_trees"][self._relation_scheme])
        roots = tree.root_edus
        self._logger.info("Root EDUs: %r.", roots)
        scores = [r.polarity_scores for r in roots]
        self._logger.info("Polarity scores of root EDUs: %r", scores)
        cls_idx = np.argmax(scores)
        self._logger.info("cls_idx: %r", cls_idx)
        self._logger.info("label: %s", IDX2CLS[cls_idx])
        self._logger.info("score: %f", scores[cls_idx])
        return IDX2CLS[cls_idx]
