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

from builtins import range
from collections import defaultdict, namedtuple
from pystruct.learners import FrankWolfeSSVM as FFSVM
from pystruct.models import EdgeFeatureLatentNodeCRF as EFLNCRF
from six import iteritems, itervalues
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np

from .constants import CLS2IDX, IDX2CLS, N_POLARITIES
from .ml import MLBaseAnalyzer
from .rst import Tree as RSTTree
from .utils import LOGGER


##################################################################
# Variables and Constants
Dataset = namedtuple("Dataset", ['X', 'Y'])
PARAM_GRID = {'C': np.linspace(0, 3, 5)}
N_FEATS = N_POLARITIES + 1


##################################################################
# Classes and Methods
def get_prnt(node1, node2, parents):
    """Determine which of the nodes is child and which is parent.

    Args:
      node1 (str): first node of the edge
      node2 (str): second node of the edge
      parents (dict): dictionary mapping nodes to their parents

    Returns:
      str, str: id's of child and parent

    """
    if parents[node1] == node2:
        return (node1, node2)
    return (node2, node1)


class InfiniteDict(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, self.__class__)

    def getitems(self, ret, pos2node, prfx=None):
        if ret is None:
            ret = []
        if prfx is None:
            prfx = []
        for k, v in iteritems(self):
            prfx.append(k)
            if isinstance(v, InfiniteDict):
                v.getitems(ret, pos2node, prfx)
            else:
                key = [-1] * len(prfx)
                for i, label in enumerate(prfx):
                    key[pos2node[i]] = label
                ret.append((tuple(key), v))
            prfx.pop()


class FrankWolfeSSVM(FFSVM):
    pass


class EdgeFeatureLatentNodeCRF(EFLNCRF):
    def marginal_loss(self, x, y, y_hat, w):
        """Compute difference between marginal probs of correct and guessed
        assignments.

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
        3-tuple : int, float, float
          number of errors, difference between the marginal distributions of
          true and predicted labelings, and the ratio of these values

        """
        n_visible = x[0].shape[0] - self._get_n_hidden(x)
        if np.all(y[:n_visible] == y_hat[:n_visible]):
            return 0, 0., 0.
        unary_potentials = np.exp(self._get_unary_potentials(x, w))
        # `pairwise_potentials` have the shape `m_edges x n_states x n_states`,
        # where the i,j-th cell in the k-th matrix represents the potential of
        # transition from label i to label j on the k-th link
        pairwise_potentials = np.exp(self._get_pairwise_potentials(x, w))
        edges = self._get_edges(x)
        n_vertices, n_states = unary_potentials.shape
        neighbors = [[] for i in range(n_vertices)]
        pairwise_weights = [[] for i in range(n_vertices)]
        # `pairwise_weights` is a list of lists, holding a transition matrix
        # for each node `i` which has an outgoing link to node `j`
        for pw, edge in zip(pairwise_potentials, edges):
            neighbors[edge[0]].append(edge[1])
            pairwise_weights[edge[0]].append(pw)
            neighbors[edge[1]].append(edge[0])
            pairwise_weights[edge[1]].append(pw.T)
        # build BFS traversal over the tree
        traversal, parents, child_cnt, pw_forward = self.top_order(
            n_vertices, n_states, neighbors, pairwise_weights
        )
        mask = np.ones(unary_potentials.shape, dtype=np.bool)
        alpha, beta, node_marginals, edge_marginals, Z = \
            self.compute_marginals(
                unary_potentials, pw_forward,
                traversal, edges, parents, child_cnt, mask
            )
        self._check_marginals(node_marginals, edge_marginals,
                              unary_potentials, pw_forward, edges,
                              traversal, parents, neighbors, mask)
        raise NotImplementedError

        _, _, Z = self.compute_alpha(unary_potentials, pairwise_potentials,
                                     neighbors, pairwise_weights, mask)
        mask[:n_visible] = 0
        mask[np.arange(n_visible), y[:n_visible]] = True
        _, Z_y = self.compute_alpha(
            unary_potentials, pairwise_potentials,
            neighbors, pairwise_weights, mask
        )
        mask[:n_visible] = 0
        mask[np.arange(n_visible), y_hat[:n_visible]] = 1
        _, Z_y_hat = self.compute_alpha(
            unary_potentials, pairwise_potentials,
            neighbors, pairwise_weights, mask
        )
        return 1, np.exp(Z_y - Z) - np.exp(Z_y_hat - Z), np.exp(Z_y - Z_y_hat)

    def top_order(self, n_vertices, n_states, neighbors, pairwise_weights):
        visited = np.zeros(n_vertices, dtype=np.bool)
        parents = -np.ones(n_vertices, dtype=np.int)
        child_cnt = np.zeros(n_vertices, dtype=np.int)
        pw_forward = np.zeros((n_vertices, n_states, n_states))
        # sort tree nodes in BFS order
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
                        child_cnt[node] += 1
                        queue.append(neighbor)
                        visited[neighbor] = True
                        pw_forward[neighbor] = pw

                    elif not parents[node] == neighbor:
                        raise ValueError("Graph not a tree")
        return traversal, parents, child_cnt, pw_forward

    def compute_marginals(self, unary_potentials, pw_forward,
                          traversal, edges, parents, child_cnt, mask):
        """Compute forward-backward scores and marginal probabilities.

        Args:

        Returns:
          np.array, float: table of (normalized) forward scores, total energy

        """
        n_vertices, n_states = unary_potentials.shape
        alpha, child_alpha, scale, Z = self.compute_alpha(
            unary_potentials, pw_forward, traversal, parents, mask
        )
        LOGGER.debug("alpha: %r", alpha)
        LOGGER.debug("child_alpha: %r", child_alpha)
        LOGGER.debug("Z: %r", Z)
        beta = self.compute_beta(
            alpha, child_alpha, scale,
            unary_potentials, pw_forward, traversal, parents, child_cnt, mask
        )
        LOGGER.debug("beta:", repr(beta))
        node_marginals, edge_marginals = self._compute_marginals(
            alpha, child_alpha, beta, scale, unary_potentials, pw_forward,
            traversal, edges, parents, child_cnt, mask
        )
        LOGGER.debug("node_marginals:", repr(node_marginals))
        LOGGER.debug("edge_marginals:", repr(edge_marginals))
        self._check_marginals(node_marginals, edge_marginals,
                              unary_potentials, pw_forward, edges,
                              traversal, parents, mask)
        return alpha, beta, node_marginals, edge_marginals, Z

    def _compute_marginals(self, alpha, chld_alpha, beta, scale,
                           unary_potentials, pw_forward, traversal, edges,
                           parents, child_cnt, mask):
        """Compute forward-backward scores and marginal probabilities.

        Args:

        Returns:
          np.array, float: table of (normalized) forward scores, total energy

        """
        # node marginals are easy
        node_marginals = alpha * beta * scale[:, None]  # None adds a new axis
        # edge marginals are a bit trickier
        n_labels = unary_potentials.shape[-1]
        edge_marginals = np.zeros((edges.shape[0], n_labels, n_labels))
        # messages from leaves to root
        for ((node1, node2), marginal) in zip(edges, edge_marginals):
            chld, prnt = get_prnt(node1, node2, parents)
            if child_cnt[prnt] > 1:
                marginal[:] = alpha[prnt]
                marginal *= scale[prnt]
                marginal /= chld_alpha[chld]
            else:
                marginal[:] = unary_potentials[prnt]
            marginal *= beta[prnt]
            LOGGER.debug("alpha[chld]: %r", alpha[chld])
            LOGGER.debug("pw_forward[chld]: %r", pw_forward[chld])
            LOGGER.debug("edge marginal: %r", marginal)
            marginal[:] = alpha[chld, None].T * pw_forward[chld] * marginal
            LOGGER.debug("* edge marginal: %r", marginal)
        return node_marginals, edge_marginals

    def compute_alpha(self, unary_potentials, pw_forward,
                      traversal, parents, mask):
        """Compute forward BP scores.

        Args:

        Returns:
          np.array, np.array, np.array, float: table of (normalized) forward
            scores, propagated messages, normalization factors, total energy

        """
        n_vertices, n_states = unary_potentials.shape
        alpha = np.ones((n_vertices, n_states))
        child_alpha = np.zeros((n_vertices, n_states))
        scale = np.zeros(n_vertices)
        # messages from leaves to the root
        for node in traversal[::-1]:
            # compute the unnormalized potential at the current node
            alpha[node] *= unary_potentials[node]
            alpha *= mask[node]
            scale[node] = np.sum(alpha[node]) or 1.
            alpha[node] /= scale[node]
            parent = parents[node]
            if parent >= 0:
                # estimate the belief
                child_alpha[node] = np.dot(alpha[node], pw_forward[node])
                # propagate belief to the parent
                alpha[parent] *= child_alpha[node]
        Z = np.sum(np.log(scale))
        return alpha, child_alpha, scale, Z

    def compute_beta(self, alpha, child_alpha, scale,
                     unary_potentials, pw_forward,
                     traversal, parents, child_cnt, mask):
        """Compute backward BP scores.

        Args:
          alpha (np.array):
          child_alpha (np.array): forward belief pro
          scale (np.array): scaling factor for alpha states
          unary_potentials (np.array): node features
          pairwise_potentials (np.array): edge features
          pairwise_weights (list[list]): pairwise potentials per node
          mask (np.array): boolean mask of allowed paths
          child_cnt (np.array): number of children for each node
          scale (np.array): energy accumulated in the forward path

        Returns:
          np.array, float: table of (normalized) forward scores, total energy

        """
        n_vertices, n_states = unary_potentials.shape
        # build BFS traversal over the tree
        beta = np.ones((n_vertices, n_states))
        # propagate messages from root to the leaves
        for node in traversal:
            crnt_beta = beta[node]
            crnt_scale = scale[node]
            # compute the unnormalized potential at the current node
            prnt = parents[node]
            # simply assign the scale factor to the root node
            if prnt >= 0:
                # if the parent has more than one children, we need to remove
                # the alpha score that from this node
                if child_cnt[prnt] > 1:
                    crnt_beta *= alpha[prnt]
                    # yes, we multiply with the parent scale
                    crnt_beta *= scale[prnt]
                    crnt_beta /= child_alpha[node]
                else:
                    crnt_beta *= unary_potentials[prnt]
                crnt_beta *= beta[prnt]
                crnt_beta[:] = np.dot(pw_forward[node], crnt_beta)
            crnt_beta /= crnt_scale
            crnt_beta *= mask[node]
        return beta

    def _check_marginals(self, node_marginals, edge_marginals,
                         unary_potentials, pw_forward, edges,
                         traversal, parents, mask):
        """Compare automatically computed marginals with brute-force estimated ones.

        """
        # compute scores for all possible tag sequences
        # place to store the scores
        tag_seq2score = InfiniteDict()
        self._check_marginals_helper(
            tag_seq2score, 1., [None] * unary_potentials.shape[0],
            traversal, 0, unary_potentials, pw_forward, parents, mask
        )
        items = []
        tag_seq2score.getitems(items,
                               {i: n for i, n in enumerate(traversal)})
        tag_seq2score = dict(items)
        Z = sum(itervalues(tag_seq2score))
        # normalize probabilities
        for tag_seq, score in iteritems(tag_seq2score):
            tag_seq2score[tag_seq] = score / Z
        # find marginal probabilities of the states
        _node_marginals = np.empty(unary_potentials.shape)
        self._compute_node_marginals(_node_marginals, tag_seq2score)
        node_marginals_sum = np.sum(node_marginals, axis=1)
        node_marginals_checksum = np.ones_like(node_marginals_sum)
        assert np.all(np.isclose(node_marginals_checksum,
                                 node_marginals_sum)), (
            "Automatically computed node marginals do not sum to 1:"
            "\n{!r} "
            ).format(node_marginals_sum)
        _node_marginals_sum = np.sum(_node_marginals, axis=1)
        assert np.all(np.isclose(node_marginals_checksum,
                                 _node_marginals_sum)), (
            "Brute-force node marginals do not sum to 1:"
            "\n{!r} "
            ).format(node_marginals_sum)
        assert np.all(np.isclose(node_marginals, _node_marginals)), (
            "Automatically and brute-force computed node marginals diverged:"
            "\n{!r}\nvs.\n{!r}"
        ).format(node_marginals, _node_marginals)
        n_labels = unary_potentials.shape[1]
        _edge_marginals = np.empty((edges.shape[0], n_labels, n_labels))
        self._compute_edge_marginals(_edge_marginals, tag_seq2score,
                                     edges, parents)
        edge_marginals_sum = np.sum(edge_marginals, axis=(1, 2))
        edge_marginals_checksum = np.ones_like(edge_marginals_sum)
        assert np.all(np.isclose(edge_marginals_checksum,
                                 edge_marginals_sum)), (
            "Automatically computed edge marginals do not sum to 1:"
                                     "\n{!r}\n{!r}"
            ).format(edge_marginals_sum, edge_marginals)
        _edge_marginals_sum = np.sum(_edge_marginals, axis=(1, 2))
        assert np.all(np.isclose(edge_marginals_checksum,
                                 _edge_marginals_sum)), (
            "Brute-force computed edge marginals do not sum to 1:"
            "\n{!r} "
            ).format(node_marginals_sum)
        assert np.all(np.isclose(edge_marginals, _edge_marginals)), (
            "Automatically and brute-force computed edge marginals diverged."
            "\n{!r}\nvs.\n{!r}"
        ).format(edge_marginals, _edge_marginals)

    def _check_marginals_helper(self, tag_seq2score, accum_score, tag_seq,
                                traversal, idx, unary_potentials, pw_forward,
                                parents, mask):
        crnt_node = traversal[idx]
        crnt_unary = unary_potentials[crnt_node] * mask[crnt_node]
        crnt_forward = pw_forward[crnt_node]
        prnt_node = parents[crnt_node]
        label_j = None
        if prnt_node >= 0:
            label_j = tag_seq[prnt_node]
        # If we have arrived at the final node, we are ready to calculate the
        # unnormalized probabilities of the whole label sequence.
        next_idx = idx + 1
        is_terminal = (next_idx >= len(traversal))
        for label_i in range(unary_potentials.shape[-1]):
            score = accum_score * crnt_unary[label_i]
            if prnt_node >= 0:
                score *= crnt_forward[label_i][label_j]
            if is_terminal:
                tag_seq2score[label_i] = score
            else:
                tag_seq[crnt_node] = label_i
                self._check_marginals_helper(
                    tag_seq2score[label_i], score, tag_seq,
                    traversal, next_idx, unary_potentials, pw_forward,
                    parents, mask
                )

    def _compute_node_marginals(self, marginals, tag_seq2score):
        n_nodes, n_labels = marginals.shape
        # iterate over each node
        for i in range(n_nodes):
            # for each node, estimate the marginal probability of each label
            for j in range(n_labels):
                marginals[i][j] = sum(
                    v for k, v in iteritems(tag_seq2score)
                    if k[i] == j
                )
        return marginals

    def _compute_edge_marginals(self, marginals, tag_seq2score,
                                edges, parents):
        n_labels = marginals.shape[-1]
        # iterate over each node
        for marginal_k, (idx1, idx2) in zip(marginals, edges):
            chld, prnt = get_prnt(idx1, idx2, parents)
            # for each node, estimate the marginal probability of each label
            for i in range(n_labels):
                for j in range(n_labels):
                    marginal_k[i][j] = sum(
                        v for k, v in iteritems(tag_seq2score)
                        if k[chld] == i and k[prnt] == j
                    )
        return marginals

    def _compute_z(self, unary_potentials, pw_forward,
                   traversal, parents, mask):
        alpha = np.ones(unary_potentials.shape)
        # messages from leaves to the root
        for node in traversal[::-1]:
            alpha[node] *= unary_potentials[node] * mask[node]
            parent = parents[node]
            if parent != -1:
                # propagate belief to the parent
                alpha[parent] *= np.dot(alpha[node], pw_forward[node])
        return alpha, np.sum(alpha[traversal[0]])


class HCRFAnalyzer(MLBaseAnalyzer):
    """Discourse-aware sentiment analysis using hidden-variable CRF.

    """
    @staticmethod
    def get_rels(forrest):
        """Extract all relations present in the forrest of RST trees.

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
