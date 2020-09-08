#!/usr/bin/env python

##################################################################
# Imports
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from scipy.special import softmax
from torch import tensor
from pyro.distributions import (
    MultivariateNormal
)
from torch.nn import Softmax
import numpy as np
import pyro


##################################################################
# Classes
class AlphaModel(PyroModule):
    def __init__(self, n_rels, n_polarities=3):
        super().__init__()
        self._n_rels = n_rels
        self._n_polarities = n_polarities
        self._softmax = Softmax(dim=-1)

    @PyroParam
    def M_Mu(self):
        Mu = np.eye(self._n_polarities, dtype="float32")
        Mu[1, 1] = 0.3
        Mu = np.tile(Mu, (self._n_rels, 1)).reshape(
                self._n_rels, self._n_polarities, self._n_polarities
        )
        return tensor(Mu)

    @PyroParam
    def M_Sigma(self):
        return tensor(np.eye(self._n_polarities, dtype="float32"))

    @PyroSample
    def M(self):
        return MultivariateNormal(self.M_Mu, self.M_Sigma)

    def _get_child_scores(self, node_scores, children, inst_indices, i,
                          n_instances, max_children):
        child_indices = children[inst_indices, i].reshape(-1)
        inst_indices.repeat(max_children, 1).t().reshape(-1)
        child_scores = node_scores[
            inst_indices.repeat(max_children, 1).t().reshape(-1),
            child_indices
        ].reshape(n_instances, max_children, -1)
        return child_scores

    def forward(self, node_scores, children, rels, labels):
        n_instances = node_scores.shape[0]
        max_depth = node_scores.shape[1]
        max_width = children.shape[-1]
        with pyro.plate("batch", size=n_instances) as inst_indices:
            print("inst_indices", inst_indices)
            for i in range(max_depth):
                print("i:", i)
                prnt_scores_i = node_scores[inst_indices, i]
                print("prnt_scores_i:", prnt_scores_i)
                rels_i = rels[inst_indices, i]
                print("rels_i:", rels_i)
                child_scores_i = self._get_child_scores(
                    node_scores, children, inst_indices, i,
                    n_instances, max_width
                )
        return self.M


a = AlphaModel(6, 3)

node_scores = tensor(softmax(np.random.uniform(0, 1, (2, 6, 3)), -1))
node_scores[0][0] = 0.
node_scores[0][1] = 0.
node_scores[1][0] = 0.
print("node_scores", node_scores)

children = np.zeros(node_scores.shape, dtype=np.int32)
children[0][-1] = [1, 2, 3]
children[1, [2, 3, 4, 5], 0] = 1
print("children", children)

rels = np.zeros(node_scores.shape, dtype=np.int32)
rels[0][-1] = [1, 2, 3]
rels[1, [2, 3, 4, 5], 0] = 2
print("rels", rels)

labels = np.array([0, 1], dtype=np.int32)
print("labels", rels)
print("a():", a(node_scores, children, rels, labels))
