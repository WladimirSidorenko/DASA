#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for all discourse-aware analyzers.

Attributes:
  BaseAnalyzer (class): abstract base for all discourse-aware sentiment
    analyzers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals
from typing import List, Optional

import numpy as np

from .base import DASBaseAnalyzer
from .constants import IDX2CLS


##################################################################
# Variables and Constants


##################################################################
# Classes
class DUSAnalyzer(DASBaseAnalyzer):
    """Main class for discourse-unaware sentiment analyzer.

    Attributes:

    """
    _name = "NoDiscourse"

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

    def predict(self, instances: List[dict],
                relation_scheme: Optional[str] = None,
                sentiment_scores: Optional[str] = None):
        """Predict label of a single input instance.

        Args:
          instances (list): input instances to classify
          relation_scheme (str): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        ret = []
        for instance_i in instances:
            scores = self._get_scores(instance_i)
            self._prune_prediction(scores)
            cls_idx = np.argmax(scores)
            ret.append(IDX2CLS[cls_idx])
        return ret

    def debug(self, instance, relation_scheme=None):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        scores = self._get_scores(instance)
        self._logger.info("Polarity scores: %r", scores)
        self._prune_prediction(scores)
        self._logger.info("Pruned polarity scores: %r", scores)
        cls_idx = np.argmax(scores)
        self._logger.info("cls_idx: %r", cls_idx)
        self._logger.info("label: %s", IDX2CLS[cls_idx])
        self._logger.info("score: %f", scores[cls_idx])
        return IDX2CLS[cls_idx]

    def _digitize_input(self, data: List[dict],
                        train_mode: bool = False) -> np.array:
        return np.array([self._get_scores(instance) for instance in data])
