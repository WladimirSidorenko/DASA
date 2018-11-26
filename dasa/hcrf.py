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
from pystruct.learners import FrankWolfeSSVM as FWSSVM
from pystruct.models import EdgeFeatureLatentNodeCRF as EFLNCRF
from six import iteritems
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from time import time
import numpy as np
import sys

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


class FrankWolfeSSVM(FWSSVM):
    def fit(self, X_train, Y_train, X_dev, Y_dev, constraints=None,
            initialize=True):
        """Learn parameters using (block-coordinate) Frank-Wolfe learning.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        contraints : ignored

        initialize : boolean, default=True
            Whether to initialize the model for the data.
            Leave this true except if you really know what you are doing.
        """
        if initialize:
            self.model.initialize(X_train, Y_train)
        self.objective_curve_, self.primal_objective_curve_ = [], []
        self.timestamps_ = [time()]
        self.w = getattr(self, "w", np.zeros(self.model.size_joint_feature))
        self.l = getattr(self, "l", 0)
        try:
            if self.batch_mode:
                self._frank_wolfe_batch(X_train, Y_train)
            else:
                self._frank_wolfe_bc(X_train, Y_train, X_dev, Y_dev)
        except KeyboardInterrupt:
            pass
        if self.verbose:
            print("Calculating final objective.")
        self.timestamps_.append(time() - self.timestamps_[0])
        self.primal_objective_curve_.append(self._objective(X_train, Y_train))
        self.objective_curve_.append(self.objective_curve_[-1])
        if self.logger is not None:
            self.logger(self, 'final')
        return self

    def _frank_wolfe_bc(self, X_train, Y_train, X_dev, Y_dev):
        """Block-Coordinate Frank-Wolfe learning.

        Compare Algorithm 3 in the reference paper.
        """
        n_samples = len(X_train)
        n_dev_samples = len(X_dev)
        # w = np.random.rand(*self.w.shape) # random initialization
        w = self.w.copy()       # zero initialization
        w_mat = np.zeros((n_samples, self.model.size_joint_feature))
        l_mat = np.zeros(n_samples)
        l = 0.0
        k = 0

        rng = check_random_state(self.random_state)
        min_dev_loss = sys.float_info.max
        for iteration in range(self.max_iter):
            if self.verbose > 0:
                print(("Iteration %d" % iteration))

            total_loss = 0.
            total_aux_loss = 0.
            perm = np.arange(n_samples)
            if self.sample_method == 'perm':
                rng.shuffle(perm)
            elif self.sample_method == 'rnd':
                perm = rng.randint(low=0, high=n_samples, size=n_samples)

            for j in range(n_samples):
                i = perm[j]
                x, y = X_train[i], Y_train[i]
                y_hat, delta_joint_feature, slack, loss, aux_loss = \
                    self.model.find_constraint(
                        x, y, w
                    )
                total_loss += loss
                total_aux_loss += aux_loss
                # ws and ls
                ws = delta_joint_feature * self.C
                ls = loss / n_samples

                # line search
                if self.line_search:
                    eps = 1e-15
                    w_diff = w_mat[i] - ws
                    gamma = ((w_diff.T.dot(w)
                             - (self.C * n_samples) * (l_mat[i] - ls))
                             / (np.sum(w_diff ** 2) + eps))
                    gamma = max(0.0, min(1.0, gamma))
                else:
                    gamma = 2.0 * n_samples / (k + 2.0 * n_samples)

                w -= w_mat[i]
                w_mat[i] = (1.0 - gamma) * w_mat[i] + gamma * ws
                w += w_mat[i]

                l -= l_mat[i]
                l_mat[i] = (1.0 - gamma) * l_mat[i] + gamma * ls
                l += l_mat[i]

                if self.do_averaging:
                    rho = 2. / (k + 2.)
                    self.w = (1. - rho) * self.w + rho * w
                    self.l = (1. - rho) * self.l + rho * l
                else:
                    self.w = w
                    self.l = l
                k += 1

            total_dev_loss = 0.
            for i in range(n_dev_samples):
                x, y = X_dev[i], Y_dev[i]
                y_hat, delta_joint_feature, slack, loss, aux_loss = \
                    self.model.find_constraint(
                        x, y, w
                    )
                total_dev_loss += loss
            if total_dev_loss < min_dev_loss:
                min_dev_loss = total_dev_loss
                best_w = self.w.copy()
            if self.verbose > 0:
                print("loss: {:f}; aux loss: {:f}; dev_loss: {:f}".format(
                    total_loss, total_aux_loss, total_dev_loss)
                )

            if self.logger is not None:
                self.logger(self, iteration)
            self.primal_objective_curve_.append(0.)
            self.objective_curve_.append(0.)
        self.w = best_w

    def _objective(self, X, Y):
        return 0.


class EdgeFeatureLatentNodeCRF(EFLNCRF):
    def find_constraint(self, x, y, w,
                        y_hat=None, relaxed=True, compute_difference=True):
        """Find most violated constraint, or, given y_hat,
        find slack and djoint_feature for this constraing.

        As for finding the most violated constraint, it is enough to compute
        joint_feature(x, y_hat), not djoint_feature, we can optionally skip
        computing joint_feature(x, y) using compute_differences=False
        """
        assert not getattr(self, 'rescale_C', False), \
            "`find_constraint` does not support rescale_C."
        assert y_hat is None, \
            "`find_constraint` does not support provided `y_hat`."
        assert compute_difference is True, \
            "`find_constraint` does not support `compute_difference == False`."
        assert not isinstance(y_hat, tuple), \
            "`find_constraint` does not support continuous loss."
        unary_potentials = np.exp(self._get_unary_potentials(x, w))
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
        traversal, parents, child_cnt, pw_forward = self.top_order(
            n_vertices, n_states, neighbors, pairwise_weights
        )
        y_prime = self.label_from_latent(y)
        self._check_size_w(w)
        y_hat = self.loss_augmented_inference(
            y_prime, unary_potentials, pw_forward, traversal, parents
        )
        delta, aux_loss, ratio = self.delta(x, y, y_hat, unary_potentials,
                                            pw_forward, traversal, edges,
                                            parents, child_cnt)
        slack = None
        loss = self.loss(y, y_hat)
        return y_hat, delta, slack, loss, aux_loss

    def loss_augmented_inference(self, y, unary_potentials,
                                 pw_forward, traversal, parents):
        self.inference_calls += 1
        n_vertices, n_states = unary_potentials.shape
        # do loss-augmentation
        unary_potentials = unary_potentials.copy()
        for l in np.arange(self.n_labels):
            # for each class, decrement features
            # for loss-agumention
            inds = np.where(y != l)[0]
            unary_potentials[inds, l] += self.class_weight[y][inds]
        mask = np.ones(unary_potentials.shape, dtype=np.bool)
        alpha, child_alpha, scale, Z = self.compute_alpha(
            unary_potentials, pw_forward, traversal, parents, mask
        )
        return np.argmax(alpha, axis=-1)

    def delta(self, x, y, y_hat, unary_potentials, pw_forward,
              traversal, edges, parents, child_cnt):
        """Compute gradient of the objective function.

        Args:
          y (np.array): correct label assignment
          y_hat (np.array): guessed abel assignment

        Retuns:
          np.array:

        """
        node_features = self._get_features(x)
        node_features = np.tile(np.expand_dims(node_features, axis=1),
                                (1, self.n_input_states, 1))
        edge_features = x[2]
        edge_features = np.repeat(edge_features,
                                  repeats=self.n_input_states ** 2,
                                  axis=1)
        delta_y, Z_y = self._delta_helper(
            y, node_features, edge_features, unary_potentials,
            pw_forward, traversal, edges, parents, child_cnt
        )
        delta_y_hat, Z_y_hat = self._delta_helper(
            y_hat, node_features, edge_features, unary_potentials,
            pw_forward, traversal, edges, parents, child_cnt
        )
        delta = ((delta_y * Z_y_hat - delta_y_hat * Z_y)
                 / np.power(Z_y_hat or 1e10, 2))
        ratio = Z_y / Z_y_hat
        aux_loss = np.maximum(1.3 - ratio, 0.)
        return delta, aux_loss, ratio

    def _delta_helper(self, y, node_features, edge_features, unary_potentials,
                      pw_forward, traversal, edges, parents, child_cnt):
        """Function for finding the derivatives and marginal prob of single assignment.

        Retuns:
          np.array, float: partial derivatives of the parameters, unnormalize
            marginal probability of the label assignment

        """
        y_mask = self.labels2mask(y)
        delta_y = np.zeros(self.size_joint_feature)
        _, _, node_marginals_y, edge_marginals_y, Z_y = self.compute_marginals(
            unary_potentials, pw_forward, traversal, edges,
            parents, child_cnt, y_mask
        )
        Z_y = np.exp(Z_y)
        # unnormalize marginals
        node_gradients = node_marginals_y * Z_y
        node_gradients = (np.expand_dims(node_gradients, axis=-1)
                          * node_features)
        node_gradients = np.ravel(node_gradients.sum(axis=0))
        n_state_features = self.n_input_states * self.n_features
        delta_y[:n_state_features] = node_gradients
        edge_marginals_y *= Z_y
        edge_gradients = edge_marginals_y.reshape(
            edge_marginals_y.shape[0], -1
        )
        edge_gradients = np.tile(edge_gradients, (1, self.n_edge_features))
        edge_gradients = edge_gradients * edge_features
        edge_gradients = edge_gradients.sum(axis=0)
        delta_y[n_state_features:] = edge_gradients
        return delta_y, Z_y

    def labels2mask(self, y):
        """Create binary mask which hides all observed labels except for that in `y`.

        Args:
          y (np.array): label sequence whose observed labels should not be hid

        Returns:
          np.array: boolean matrix which hides all observed labels except for
            the ones

        """
        mask = np.ones((len(y), self.n_states), dtype=np.bool)
        # determine indices of observed states
        observed_states = np.nonzero(y < self.n_labels)
        observed_labels = y[observed_states]
        mask[observed_states] = False
        mask[observed_states, observed_labels] = True
        return mask

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
        LOGGER.debug("unary potentials: %r", unary_potentials)
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
        LOGGER.debug("beta: %r", beta)
        node_marginals, edge_marginals = self._compute_marginals(
            alpha, child_alpha, beta, scale, unary_potentials, pw_forward,
            traversal, edges, parents, child_cnt
        )
        LOGGER.debug("node_marginals: %r", node_marginals)
        LOGGER.debug("edge_marginals: %r", edge_marginals)
        return alpha, beta, node_marginals, edge_marginals, Z

    def _compute_marginals(self, alpha, chld_alpha, beta, scale,
                           unary_potentials, pw_forward, traversal, edges,
                           parents, child_cnt):
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
            alpha[node] *= mask[node]
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

    def __init__(self, relation_scheme, marginalized, *args, **kwargs):
        """Class constructor.

        Args:
          relation_scheme (str): relations to use
          marginalized (bool): use marginal CRFs
          args (list[str]): arguments to use for initializing models
          kwargs (dict): keyword arguments to use for initializing models

        """
        super(HCRFAnalyzer, self).__init__(*args, **kwargs)
        self._name = "HCRF"
        self._relation_scheme = relation_scheme
        self._marginalized = marginalized
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
        self._model.fit(train_set[0], train_set[1], dev_set[0], dev_set[1])
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
            if self._marginalized:
                model_cls = EdgeFeatureLatentNodeCRF
                optimizer_cls = FrankWolfeSSVM
                max_iter = 100
                C = 0.3
            else:
                model_cls = EFLNCRF
                optimizer_cls = FWSSVM
                max_iter = 1000
                C = 1.05
            model = model_cls(n_labels=N_POLARITIES,
                              n_features=N_FEATS,
                              n_edge_features=self._n_rels,
                              n_hidden_states=N_POLARITIES,
                              latent_node_features=True,
                              inference_method="max-product")
            # best C: 1.05 on PotTS and 1.05 on SB10k
            self._model = optimizer_cls(
                model=model, C=C, max_iter=max_iter, verbose=1
            )
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
