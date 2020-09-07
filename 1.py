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

    def forward(self, node_scores, children, rels, labels):
        n_instances = node_scores.shape[0]
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
# print(a())
