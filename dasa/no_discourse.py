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
from typing import Optional

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
    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(DUSAnalyzer, self).__init__(*args, **kwargs)
        self._name = "NoDiscourse"

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

    def predict(self, instance: dict,
                relation_scheme: Optional[str] = None,
                sentiment_scores: Optional[str] = None):
        """Predict label of a single input instance.

        Args:
          instance (dict): input instance to classify
          relation_scheme (str): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        scores = instance["polarity_scores"]
        cls_idx = np.argmax(scores)
        return IDX2CLS[cls_idx]

    def debug(self, instance, relation_scheme=None):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        scores = instance["polarity_scores"]
        self._logger.info("Polarity scores: %r", scores)
        cls_idx = np.argmax(scores)
        self._logger.info("cls_idx: %r", cls_idx)
        self._logger.info("label: %s", IDX2CLS[cls_idx])
        self._logger.info("score: %f", scores[cls_idx])
        return IDX2CLS[cls_idx]
