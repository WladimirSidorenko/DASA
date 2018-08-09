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


##################################################################
# Variables and Constants


##################################################################
# Classes
class DASLastAnalyzer(DASBaseAnalyzer):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          a_args (list[str]): arguments to use for initializing models
          a_kwargs (dict): keyword arguments to use for initializing models

        """
        super(DASLastAnalyzer, self).__init__(*args, **kwargs)
        self._name = "LastEDU"

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
        return IDX2CLS[np.argmax(instance["edus"][-1]["polarity_scores"])]

    def debug(self, instance):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        print("Polarity scores of all EDUs: %r.",
              [edu["polarity_scores"] for edu in instance["edus"]])
        self._logger.info("Polarity scores of last EDU: %r",
                          instance["edus"][-1]["polarity_scores"])
        return IDX2CLS[np.argmax(instance["edus"][-1]["polarity_scores"])]
