#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for predicting polarity of a tweet using
hidden-variable CRF.

Attributes:
  HCRFAnalyzer (class): class for predicting polarity of a tweet using
    hidden-variable CRF

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from collections import namedtuple
from pystruct.learners import FrankWolfeSSVM as FFSVM
from pystruct.models import EdgeFeatureLatentNodeCRF as EFLNCRF
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np

from .constants import CLS2IDX, IDX2CLS, N_POLARITIES
from .ml import MLBaseAnalyzer
from .rst import Tree as RSTTree


##################################################################
# Variables and Constants
Dataset = namedtuple("Dataset", ['X', 'Y'])
PARAM_GRID = {'C': np.linspace(0, 3, 5)}
N_FEATS = N_POLARITIES + 1


##################################################################
# Class
class FrankWolfeSSVM(FFSVM):
    pass


class EdgeFeatureLatentNodeCRF(EFLNCRF):
    def marginal_loss(self, x, y, y_hat, w):
        """Compute the difference between marginal probs of correct and guessed
        labelings.

        Parameters
        ----------
        x : tuple
            Unary evidence.

        y : ndarray
            true labeling for x

        y_hat : ndarray
            predicted labeling for x

        Returns
        -------
        d : float
          difference between the marginal distributions of true and predicted
          labelings

        """
        if self.label_from_latent(y) == self.label_from_latent(y_hat):
            return 0.
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        mask = np.ones(unary_potentials.shape, dtype=np.bool)
        _, Z = self.compute_alpha(unary_potentials, pairwise_potentials,
                                  edges, mask)
        n_visible = unary_potentials.shape[0] - self._get_n_hidden(x)
        mask[:n_visible] = 0
        mask[np.arange(n_visible), y[:n_visible]] = True
        _, Z_y = self.compute_alpha(
            unary_potentials, pairwise_potentials, edges, mask
        )
        mask[:n_visible] = 0
        mask[np.arange(n_visible), y_hat[:n_visible]] = 1
        _, Z_y_hat = self.compute_alpha(
            unary_potentials, pairwise_potentials, edges, mask
        )
        return np.exp(Z_y - Z) - np.exp(Z_y_hat - Z)

    def compute_alpha(self, unary_potentials, pairwise_potentials,
                      edges, mask):
        """Compute forward BP scores.

        Args:
          unary_potentials (np.array): node features
          pairwise_potentials (np.array): edge features
          edges (list): list of edges
          mask (np.array): boolean mask of allowed paths

        Returns:
          np.array, float: table of (normalized) forward scores, total energy

        """
        unary_potentials = np.exp(unary_potentials)
        pairwise_potentials = np.exp(pairwise_potentials)
        n_vertices, n_states = unary_potentials.shape
        neighbors = [[] for i in range(n_vertices)]
        pairwise_weights = [[] for i in range(n_vertices)]
        for pw, edge in zip(pairwise_potentials, edges):
            neighbors[edge[0]].append(edge[1])
            pairwise_weights[edge[0]].append(pw)
            neighbors[edge[1]].append(edge[0])
            pairwise_weights[edge[1]].append(pw.T)

        # build BFS traversal over the tree
        traversal, parents, pw_forward = self.top_order(
            n_vertices, n_states, neighbors, pairwise_weights
        )
        alpha = np.ones((n_vertices, n_states))
        scale = np.zeros(n_vertices)
        # messages from leaves to the root
        for node in traversal[::-1]:
            # compute the unnormalized potential at the current node
            alpha[node] *= unary_potentials[node]
            alpha *= mask[node]
            scale[node] = np.sum(alpha[node])
            alpha[node] /= scale[node]
            parent = parents[node]
            if parent != -1:
                # propagate belief to the parent
                alpha[parent] *= np.dot(alpha[node], pw_forward[node])
        Z = np.sum(np.log(scale))
        return alpha, Z

    def _compute_z(self, unary_potentials, pairwise_potentials, edges):
        unary_potentials = np.exp(unary_potentials)
        pairwise_potentials = np.exp(pairwise_potentials)
        n_vertices, n_states = unary_potentials.shape
        neighbors = [[] for i in range(n_vertices)]
        pairwise_weights = [[] for i in range(n_vertices)]
        for pw, edge in zip(pairwise_potentials, edges):
            neighbors[edge[0]].append(edge[1])
            pairwise_weights[edge[0]].append(pw)
            neighbors[edge[1]].append(edge[0])
            pairwise_weights[edge[1]].append(pw.T)

        # build BFS traversal over the tree
        traversal, parents, pw_forward = self.top_order(
            n_vertices, n_states, neighbors, pairwise_weights
        )
        alpha = np.ones((n_vertices, n_states))
        # messages from leaves to the root
        for node in traversal[::-1]:
            alpha[node] *= unary_potentials[node]
            parent = parents[node]
            if parent != -1:
                # propagate belief to the parent
                alpha[parent] *= np.dot(alpha[node], pw_forward[node])
        return np.sum(alpha[traversal[0]])

    def top_order(self, n_vertices, n_states, neighbors, pairwise_weights):
        visited = np.zeros(n_vertices, dtype=np.bool)
        parents = -np.ones(n_vertices, dtype=np.int)
        pw_forward = np.zeros((n_vertices, n_states, n_states))
        # build a breadth first search of the tree
        traversal = []
        lonely = 0
        while lonely < n_vertices:
            for i in range(lonely, n_vertices):
                if not visited[i]:
                    queue = [i]
                    lonely = i + 1
                    visited[i] = True
                    break
                lonely = n_vertices

            while queue:
                node = queue.pop(0)
                traversal.append(node)
                for pw, neighbor in zip(pairwise_weights[node],
                                        neighbors[node]):
                    if not visited[neighbor]:
                        parents[neighbor] = node
                        queue.append(neighbor)
                        visited[neighbor] = True
                        pw_forward[neighbor] = pw

                    elif not parents[node] == neighbor:
                        raise ValueError("Graph not a tree")
        return traversal, parents, pw_forward


class HCRFAnalyzer(MLBaseAnalyzer):
    """Discourse-aware sentiment analysis using hidden-variable CRF.

    """
    @staticmethod
    def get_rels(forrest):
        """Extract all relations present in forrest of RST trees.

        Args:
          forrest (list[rst.Tree]): list of RST trees

        """
        rels = set()
        for tree in forrest:
            nodes = [tree]
            while nodes:
                node = nodes.pop(0)
                nodes.extend(node.children)
                rels.add(node.rel2par)
        return rels

    def __init__(self, relation_scheme, *args, **kwargs):
        """Class constructor.

        Args:
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(HCRFAnalyzer, self).__init__(*args, **kwargs)
        self._name = "HCRF"
        self._relation_scheme = relation_scheme
        self._model = None

    def _train(self, train_set, dev_set, grid_search=True, balance=False):
        def score(y_gold, y_pred):
            return f1_score([y[0] for y in y_gold],
                            [y[0] for y in y_pred], average="macro")

        if grid_search:
            def cv_scorer(estimator, X_test, y_test):
                return score(y_test, estimator.predict(X_test))

            self._model = GridSearchCV(self._model, PARAM_GRID,
                                       scoring=cv_scorer)
        self._model.fit(*train_set)
        w = self._model.w
        lncrf = self._model.model
        unary_params = w[:lncrf.n_input_states * lncrf.n_features].reshape(
            lncrf.n_input_states, lncrf.n_features)
        self._logger.info("unary params: %r", unary_params)
        pairwise_params = w[lncrf.n_input_states * lncrf.n_features:]
        self._logger.info("pairwise params: %r", pairwise_params)
        if grid_search:
            cv_results = self._model.cv_results_
            for mean, std, params in zip(cv_results["mean_test_score"],
                                         cv_results["std_test_score"],
                                         cv_results["params"]):
                self._logger.info("CV results: %f (+/-%f) (%r)",
                                  mean, std, params)
            self._logger.info("Best parameters: %s", self._model.best_params_)
        dev_macro_f1 = score(dev_set.Y, self._model.predict(dev_set.X))
        self._logger.info("Macro F1-score on dev set: %.2f", dev_macro_f1)

    def predict(self, instance):
        tree = RSTTree(instance,
                       instance["rst_trees"][self._relation_scheme]).to_deps()
        x, _ = self._digitize_instance(instance, tree, train_mode=False)
        cls_idx = self._model.predict([x])[0][0]
        return IDX2CLS[cls_idx]

    def debug(self, instance):
        tree = RSTTree(instance,
                       instance["rst_trees"][self._relation_scheme]).to_deps()
        self._logger.debug("instance: %r", instance)
        self._logger.debug("tree: %r", tree)
        x, _ = self._digitize_instance(instance, tree, train_mode=False)
        self._logger.debug("features: %r", x)
        cls_idx = self._model.predict([x])[0][0]
        cls = IDX2CLS[cls_idx]
        self._logger.debug("cls_idx: %r (%s)", cls_idx, cls)
        model = self._model.model
        w = self._model.w
        unary_potentials = model._get_unary_potentials(x, w)
        self._logger.debug("unary_potentials: %r", unary_potentials)
        pairwise_potentials = model._get_pairwise_potentials(x, w)
        self._logger.debug("pairwise_potentials: %r",
                           pairwise_potentials)
        edges = model._get_edges(x)
        mask = np.ones(unary_potentials.shape, dtype=np.bool)
        alpha, Z = model.compute_alpha(
            unary_potentials, pairwise_potentials, edges, mask
        )
        self._logger.debug("alpha: %r", alpha)
        self._logger.debug("Z: %r", Z)
        return cls

    def _digitize_data(self, data, train_mode=False):
        n = len(data)
        dataset = Dataset([None] * n, [None] * n)
        forrest = [
            RSTTree(instance,
                    instance["rst_trees"][self._relation_scheme]).to_deps()
            for instance in data
        ]
        if train_mode:
            self._rel2idx = {
                rel_i: i
                for i, rel_i in enumerate(set(
                        [rel
                         for tree in forrest
                         for rel in self.get_rels(tree)]
                ))
            }

            self._n_rels = len(self._rel2idx)
            model = EdgeFeatureLatentNodeCRF(n_labels=N_POLARITIES,
                                             n_features=N_FEATS,
                                             n_edge_features=self._n_rels,
                                             n_hidden_states=N_POLARITIES,
                                             latent_node_features=True,
                                             inference_method="max-product")
            # best C: 1.05 on PotTS and 1.05 on SB10k
            self._model = FrankWolfeSSVM(model=model, C=1.05, max_iter=1000,
                                         verbose=1)
            # we use `_restore` to set up the model's logger
            self._restore(None)

        for i, (instance_i, tree_i) in enumerate(zip(data, forrest)):
            dataset.X[i], dataset.Y[i] = self._digitize_instance(
                instance_i, tree_i, train_mode=True
            )
        return dataset

    def _digitize_instance(self, instance, tree, train_mode=True):
        n_edus = len(instance["edus"])
        feats = np.zeros((n_edus + 1, N_FEATS), dtype=np.float32)
        feats[0, :-1] = instance["polarity_scores"]
        feats[0, -1] = 1
        for i, edu_i in enumerate(instance["edus"], 1):
            feats[i, :-1] = edu_i["polarity_scores"]
            feats[i, -1] = 1
        self._logger.debug("feats: %r", feats)
        self._logger.debug("tree: %r", tree)
        edges = np.zeros((len(tree) - 1, 2), dtype=np.uint8)
        edge_feats = np.zeros((len(tree) - 1, self._n_rels))
        i = 0
        for node in tree:
            if node.parent is not None:
                edges[i, 0] = node.parent.id + 1
                edges[i, 1] = node.id + 1
                edge_idx = self._rel2idx[node.rel2par]
                edge_feats[i, edge_idx] = 1
                i += 1
        if train_mode:
            labels = np.argmax(feats[:, :-1], axis=1) + N_POLARITIES
            labels[0] = CLS2IDX[instance["label"]]
        else:
            labels = None
        self._logger.debug("edge_feats: %r", edge_feats)
        self._logger.debug("labels: %r", labels)
        return ((feats, edges, edge_feats, n_edus), labels)

    def _reset(self):
        super(HCRFAnalyzer, self)._reset()
        if isinstance(self._model, GridSearchCV):
            self._model.estimator._logger = None
            self._model.scoring = None
            self._model.scorer_ = None
        else:
            self._model._logger = None

    def _restore(self, a_path):
        if a_path is not None:
            super(HCRFAnalyzer, self)._restore(a_path)

        def logger(x, *args, **kwargs):
            self._logger.debug(*args, **kwargs)

        if isinstance(self._model, GridSearchCV):
            self._model.estimator._logger = logger
        else:
            self._model._logger = logger
