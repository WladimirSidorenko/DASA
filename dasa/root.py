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
from functools import partial
from typing import Optional

import numpy as np

from .base import DASBaseAnalyzer
from .constants import IDX2CLS
from .rst import Tree as RSTTree


##################################################################
# Variables and Constants


##################################################################
# Classes
class RootAnalyzer(DASBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    _name = "RootEDU"

    def __init__(self, relation_scheme: str, sentiment_scores: str,
                 n_classes: int):
        """Class constructor.

        Args:

        """
        super().__init__(sentiment_scores, n_classes)
        self.relation_scheme = relation_scheme

    def train(self, *args, **kwargs):
        """Train specified model(s) on the provided data.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        Returns:
          void:

        """
        # this analyzer does not require training
        pass

    def fit(self, *args, **kwargs):
        """Analogue of `train` method applied directly to features.

        """
        pass

    def predict_instance(self, instance: dict,
                         relation_scheme: Optional[str] = None,
                         sentiment_scores: Optional[str] = None):
        if relation_scheme is None:
            relation_scheme = self.relation_scheme
        get_scores = partial(self._get_scores,
                             scores_key=sentiment_scores)
        tree = RSTTree(instance,
                       instance["rst_trees"][relation_scheme],
                       get_scores)
        scores = np.sum(r.polarity_scores for r in tree.root_edus)
        self._prune_prediction(scores)
        return IDX2CLS[np.argmax(scores)]

    def debug(self, instance, relation_scheme=None):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label
          relation_scheme (str): relation scheme to use

        Note:
          modifies input tweet in place

        """
        if relation_scheme is None:
            relation_scheme = self.relation_scheme
        tree = RSTTree(instance, instance["rst_trees"][self.relation_scheme])
        roots = tree.root_edus
        self._logger.info("Root EDUs: %r.", roots)
        scores = [r.polarity_scores for r in roots]
        self._logger.info("Polarity scores of root EDUs: %r", scores)
        cls_idx = np.argmax(scores)
        self._logger.info("cls_idx: %r", cls_idx)
        self._logger.info("label: %s", IDX2CLS[cls_idx])
        self._logger.info("score: %f", scores[cls_idx])
        return IDX2CLS[cls_idx]
