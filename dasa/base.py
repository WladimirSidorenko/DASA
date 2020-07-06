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
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from typing import List, Optional
try:
    from cPickle import dump, load
except ImportError:
    from _pickle import dump, load

import abc
import numpy as np
import os

from .rst import RSTTree
from .utils import LOGGER


##################################################################
# Variables and Constants
DEBUG_NFEATS = 10


##################################################################
# Classes
class DASBaseAnalyzer(BaseEstimator):
    """Main class for coarse-grained sentiment analyzer.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta
    _name = "BaseAnalyzer"

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

    def __init__(self, sentiment_scores: str, n_classes: int):
        """Class constructor.

        Args:
          sentiment_scores (str): key of sentiment scores
          n_classes (int): number of sentiment classes to predict
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        self.sentiment_scores = sentiment_scores
        self.n_classes = n_classes
        self._wbench = None
        self._logger = LOGGER

    def cross_validate(self, X: List[dict]):
        """Cross-validate specified model(s) on the provided data.

        Args:
          X (List[dict]): data to cross-validate the classifier on

        Returns:
          void:

        """
        scorers = ("precision_macro", "recall_macro",
                   "f1_macro", "accuracy")
        Y = self._digitize_labels(X)
        results = cross_validate(self, X, Y, scoring=scorers)
        for scorer_i in scorers:
            stat = results["test_" + scorer_i]
            if scorer_i == "accuracy":
                stat *= 100
            print("{:>10}: {:.4f} (+/- {:.2f})".format(
                scorer_i, np.mean(stat), np.std(stat)))

    def build_rst(self, instance: dict,
                  relation_scheme: Optional[str] = None,
                  scores_key: Optional[str] = None) -> RSTTree:
        """Construct RSTTree for the given instance.

        """
        if relation_scheme is not None:
            tree = instance["rst_trees"][relation_scheme]
        elif hasattr(self, "relation_scheme"):
            tree = instance["rst_trees"][self.relation_scheme]
        else:
            tree = instance["rst_trees"]
        sentiment_scores = [self._get_scores(edu_i, scores_key)
                            for edu_i in instance["edus"]]
        return RSTTree(instance, tree, sentiment_scores)

    @abc.abstractmethod
    def _digitize_input(self, data: List[dict],
                        train_mode: bool = False) -> np.array:
        raise NotImplementedError

    def _digitize_labels(self, data: List[dict]) -> np.array:
        return np.array([instance["label"] for instance in data])

    @abc.abstractmethod
    def train(self, train_set, dev_set=None,
              grid_search=True, balance=False):
        """Train specified model(s) on the provided data.

        Args:
          train_set (list or None):
            training set
          dev_set (list or None):
            development set
          grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          balance (bool): balance dataset to get equal number of instances
            for all classes (via downsampling)

        Returns:
          void:

        """
        raise NotImplementedError

    def predict(self, instances: List[dict],
                relation_scheme: Optional[str] = None,
                sentiment_scores: Optional[str] = None) -> List[str]:
        """Predict label of a single input instance.

        Args:
          instance (list): input instances to classify
          relation_scheme (str): relation scheme to use
          sentiment_scores (str): key of sentiment scores to use

        Returns:
          list[str]: predicted labels

        """
        ret = []
        for instance_i in instances:
            ret.append(self.predict_instance(instance_i, relation_scheme,
                                             sentiment_scores))
        return ret

    @abc.abstractmethod
    def predict_instance(self, instance: dict,
                         relation_scheme: Optional[str] = None,
                         sentiment_scores: Optional[str] = None) -> str:
        """Predict label for a single input instance.

        Args:
          instance (list): input instances to classify
          relation_scheme (str): relation scheme to use
          sentiment_scores (str): base sentiment scores to use

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        raise NotImplementedError

    @abc.abstractmethod
    def debug(self, instance, relation_scheme=None):
        """Explain predictions of each classifier.

        Args:
          instance (dict): input instance to classify
          relation_scheme (str): relation scheme to use
          sentiment_scores (str): base sentiment scores to use

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

    def _get_scores(self, item: dict,
                    scores_key: Optional[str] = None) -> np.array:
        """Obtain base sentiment scores for EDU or whole document.

        Args:
          item (dict): EDU or whole document
          scores_key (dict): symbolic key of base sentiment scores

        Returns:
          np.array: sentiment scores of analyzed item

        """
        if scores_key is None:
            scores_key = self.sentiment_scores
        return np.array(item["polarity_scores"][scores_key])

    def _prune_prediction(self, scores: np.array) -> np.array:
        """Remove.

        Args:
          item (dict): EDU or whole document
          scores_key (dict): symbolic key of base sentiment scores

        Returns:
          np.array: sentiment scores of analyzed item

        """
        n_predicted = scores.shape[-1]
        if n_predicted == 3 and self.n_classes == 2:
            neut_scores = scores[1] / 2.
            scores[1] = 0
            scores[0] += neut_scores
            scores[2] += neut_scores
        return scores

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
