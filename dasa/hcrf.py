#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet using
hidden-variable CRF.

Attributes:
  XXXAnalyzer (class): class for predicting polarity of a tweet using
    hidden-variable CRF

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from collections import namedtuple
from pystruct.learners import FrankWolfeSSVM
from pystruct.models import LatentNodeCRF
from sklearn.metrics import f1_score
import numpy as np

from .constants import CLS2IDX, IDX2CLS, N_POLARITIES
from .ml import MLBaseAnalyzer
from .rst import Tree as RSTTree


##################################################################
# Variables and Constants
Dataset = namedtuple("Dataset", ['X', 'Y'])


##################################################################
# Class
class HCRFAnalyzer(MLBaseAnalyzer):
    """Discourse-aware sentiment analysis using hidden-variable CRF.

    """

    def __init__(self, relation_scheme, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(HCRFAnalyzer, self).__init__(*args, **kwargs)
        self._name = "HCRF"
        self._relation_scheme = relation_scheme
        model = LatentNodeCRF(n_labels=N_POLARITIES,
                              n_features=N_POLARITIES,
                              n_hidden_states=N_POLARITIES,
                              latent_node_features=True)

        self._model = FrankWolfeSSVM(model=model, C=.1, max_iter=100)
        # we use `_restore` to set up the model's logger
        self._restore(None)

    def _train(self, train_set, dev_set):
        self._model.fit(*train_set)
        y_pred = [y[0]
                  for y in self._model.predict(dev_set.X)]
        dev_macro_f1 = f1_score([y[0] for y in dev_set.Y], y_pred,
                                average="macro")
        self._logger.info("Macro F1-score on dev set: %.2f", dev_macro_f1)

    def predict(self, instance):
        x, _ = self._digitize_instance(instance)
        cls_idx = self._model.predict([x])[0][0]
        return IDX2CLS[cls_idx]

    def debug(self, instance):
        self._logger.debug("instance: %r", instance)
        x, _ = self._digitize_instance(instance)
        self._logger.debug("features: %r", x)
        cls_idx = self._model.predict([x])[0][0]
        cls = IDX2CLS[cls_idx]
        self._logger.debug("cls_idx: %r (%s)", cls_idx, cls)
        return cls

    def _digitize_data(self, data, train_mode=False):
        n = len(data)
        dataset = Dataset([None] * n, [None] * n)
        for i, instance_i in enumerate(data):
            dataset.X[i], dataset.Y[i] = self._digitize_instance(instance_i)
        return dataset

    def _digitize_instance(self, instance, train_mode=True):
        self._logger.debug("instance: %r", instance)
        n_edus = len(instance["edus"])
        feats = np.zeros((1 + n_edus, N_POLARITIES), dtype=np.float32)
        feats[0, :] = instance["polarity_scores"]
        for i, edu_i in enumerate(instance["edus"], 1):
            feats[i, :] = edu_i["polarity_scores"]
        self._logger.debug("feats: %r", feats)
        tree = RSTTree(instance,
                       instance["rst_trees"][self._relation_scheme]).to_deps()
        self._logger.debug("tree: %r", tree)
        edges = np.zeros((len(tree) - 1, 2), dtype=np.uint8)
        i = 0
        for node in tree:
            if node.parent is not None:
                edges[i, 0] = node.parent.id + 1
                edges[i, 1] = node.id + 1
                i += 1
        self._logger.debug("edges: %r", edges)
        if train_mode:
            labels = np.argmax(feats, axis=1)
            labels[0] = CLS2IDX[instance["label"]]
        else:
            labels = None
        return ((feats, edges, n_edus), labels)

    def _reset(self):
        super(HCRFAnalyzer, self)._reset()
        self._model._logger = None

    def _restore(self, a_path):
        if a_path is not None:
            super(HCRFAnalyzer, self)._restore(a_path)

        def logger(x, *args, **kwargs):
            self._logger.debug(*args, **kwargs)

        self._model._logger = logger
