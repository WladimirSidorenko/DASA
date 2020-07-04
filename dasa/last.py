#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet based on the
    polarity of the last EDU.

Attributes:
  DASLastAnalyzer (class): class for predicting polarity of a tweet based on
    the polarity of the last EDU

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
class LastAnalyzer(DASBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    _name = "LastEDU"

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
        scores = self._get_scores(instance["edus"][-1])
        self._prune_prediction(scores)
        cls_idx = np.argmax(scores)
        return IDX2CLS[cls_idx]

    def debug(self, instance, relation_scheme=None):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify
          relation_scheme (str): relation scheme to use

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        print("Polarity scores of all EDUs: %r.",
              [edu["polarity_scores"] for edu in instance["edus"]])
        self._logger.info("Polarity scores of the last EDU: %r",
                          instance["edus"][-1]["polarity_scores"])
        cls_idx = np.argmax(instance["edus"][-1]["polarity_scores"])
        self._logger.info("cls_idx: %r", cls_idx)
        self._logger.info("label: %s", IDX2CLS[cls_idx])
        self._logger.info("score: %f",
                          instance["edus"][-1]["polarity_scores"][cls_idx])
        return IDX2CLS[cls_idx]
