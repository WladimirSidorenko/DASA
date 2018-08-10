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
try:
    from cPickle import dump, load
except ImportError:
    from _pickle import dump, load

import abc
import os

from .constants import CLS2IDX
from .utils import LOGGER, balance_data


##################################################################
# Variables and Constants
DEBUG_NFEATS = 10


##################################################################
# Classes
class DASBaseAnalyzer(object):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def load(a_path):
        """Load serialized model from disc.

        Args:
          a_path (str): path to file from which to load the model

        Returns:
          (DASentimentAnalyzer)

        """
        LOGGER.debug("Loading model from file: %s", a_path)
        with open(a_path, "rb") as ifile:
            analyzer = load(ifile)
        analyzer._restore(a_path)
        return analyzer

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          a_args (list[str]): arguments to use for initializing models
          a_kwargs (dict): keyword arguments to use for initializing models

        """
        self._name = "BaseAnalyzer"
        self._n_cls = 0
        self._model = None
        self._wbench = None
        self._logger = LOGGER

    def train(self, a_train_data, a_dev_data=None,
              a_grid_search=True, a_balance=False):
        """Train specified model(s) on the provided data.

        Args:
          a_train_data (list or None):
            training set
          a_dev_data (list or None):
            development set
          a_grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          a_balance (bool): balance dataset to get equal number of instances
            for all classes (via downsampling)

        Returns:
          void:

        """
        # convert polarity classes to integer indices
        self._n_cls = len(
            set(t.label for t in a_train_data) |
            set(t.label for t in (a_dev_data if a_dev_data else []))
        )
        train_x, train_y = self._generate_ts(a_train_data)
        dev_x, dev_y = self._generate_ts(a_dev_data)
        # separately train and dump each model
        if a_balance:
            train_x, train_y = balance_data(train_x, train_y)
        LOGGER.debug("Training model...")
        self._model.train(train_x, train_y, dev_x, dev_y,
                          a_grid_search=a_grid_search)
        LOGGER.debug("Model trained...")

    @abc.abstractmethod
    def predict(self, instance):
        """Predict label of a single input instance.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        raise NotImplementedError

    @abc.abstractmethod
    def debug(self, instance):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        raise NotImplementedError

    def save(self, a_path):
        """Dump model to disc.

        Args:
          a_models (list[str]): type of the models to train

        """
        dirname = self._check_path(a_path)
        # store each trained model
        logger = self._logger
        self._reset()
        logger.debug("Saving analyzer %s in %s.", self._name, dirname)
        with open(a_path, "wb") as ofile:
            dump(self, ofile)
        logger.debug("Analyzer %s saved", self._name)

    def _check_path(self, a_path):
        """Check whether path can be created and is writable.

        Args:
          a_path (str): file path to be checked

        Returns:
          str: directory name of the path

        """
        dirname = os.path.dirname(a_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif not os.path.exists(a_path):
            if (not os.access(dirname, os.W_OK) or not os.path.isdir(dirname)):
                raise RuntimeError("Cannot write to directory '{:s}'.".format(
                    dirname))
        else:
            if not os.access(a_path, os.W_OK):
                raise RuntimeError("Cannot write to file '{:s}'.".format(
                    a_path))
        return dirname

    def _generate_ts(self, a_data):
        """Generate training set.

        Args:
          a_data (list): input instances

        Returns:
          2-tuple(list, list):
            lists of input features and expected classes

        """
        x, y = [], []
        if not a_data:
            return (x, y)
        for msg_i in a_data:
            if not msg_i:
                continue
            x.append(msg_i)
            y_i = msg_i.label
            # we use a pre-defined mapping of symbolic labels to integers, as
            # we need these labels to be sorted proportionally to the
            # subjective scores they get assigned when optimizing threshold of
            # lexicon-based methods
            assert y_i in CLS2IDX, "Unknown label {:s}".format(y_i)
            y_i = CLS2IDX[y_i]
            y.append(y_i)
        return (x, y)

    def _reset(self):
        """Remove members which cannot be serialized.

        """
        self._logger = None

    def _restore(self, a_path):
        """Restore members which could not be serialized.

        """
        # normalize paths to the serialized models
        self._dirname = os.path.dirname(a_path)
        self._logger = LOGGER
