#!/usr/bin/env python

##################################################################
# Imports
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from scipy.special import softmax
from torch import tensor
from pyro.distributions import (MultivariateNormal)
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
        # Mu[1, 1] = 0.3
        Mu = np.tile(Mu, (self._n_rels, 1)).reshape(
                self._n_rels, self._n_polarities, 1, self._n_polarities
        )
        print("Mu", Mu, Mu.shape)
        Mu *= np.arange(self._n_rels).reshape(self._n_rels, 1, 1, 1)
        print("*Mu", Mu, Mu.shape)
        return tensor(Mu)

    @PyroParam
    def M_Sigma(self):
        Sigma = np.eye(self._n_polarities, dtype="float32")
        print("Sigma", Sigma, Sigma.shape)
        return tensor(Sigma)

    @PyroSample
    def M(self):
        M = MultivariateNormal(self.M_Mu, self.M_Sigma)
        print("M", M, list(M.batch_shape))
        print("M.batch_shape", M.batch_shape)
        print("M.event_shape", M.event_shape)
        return M

    def _get_child_scores(self, node_scores, children, inst_indices, i,
                          n_instances, max_children):
        child_indices = children[inst_indices, i].reshape(-1)
        print("child_indices:", child_indices)
        child_scores = node_scores[
            inst_indices.repeat(max_children, 1).t().reshape(-1),
            child_indices
        ].reshape(n_instances, max_children, -1)
        print("* child_indices:", child_indices)
        return child_scores

    def _forward_node(self, var_sfx: str, prnt_probs: tensor,
                      child_probs, rels):
        # The vector `nz_chld_indices` will contain indices of the child_probs
        # which are not zero.
        nz_chld_indices = child_probs.sum(dim=-1).nonzero().squeeze(-1)
        print("nz_chld_indices:", nz_chld_indices)
        if nz_chld_indices.nelement() == 0:
            return None, None, None, None
        # Only leave `parent`, `child`, and `rels` elements for which
        # `child_probs` are not zero.
        child_probs = child_probs[nz_chld_indices]
        print("* child_probs:", child_probs)
        rels = rels[nz_chld_indices]
        print("nz_rels:", rels)
        print("M:", self.M, self.M.shape)
        M_nz_rels = self.M[rels, :, nz_chld_indices, :].squeeze(1)
        print("M[nz_rels]:", M_nz_rels, M_nz_rels.shape)
        raise NotImplementedError

    def forward(self, node_scores, children, rels, labels):
        n_instances = node_scores.shape[0]
        max_depth = node_scores.shape[1]
        max_width = children.shape[-1]
        with pyro.plate("batch", size=n_instances) as inst_indices:
            print("inst_indices", inst_indices)
            for i in range(max_depth):
                print("i:", i)
                prnt_scores_i = node_scores[inst_indices, i]
                print("prnt_scores[{}]:".format(i), prnt_scores_i)
                rels_i = rels[inst_indices, i]
                print("rels[i]:".format(i), rels_i)
                child_scores_i = self._get_child_scores(
                    node_scores, children, inst_indices, i,
                    n_instances, max_width
                )
                print("child_scores[{}]:".format(i), child_scores_i)
                for j in range(max_width):
                    child_scores_ij = child_scores_i[inst_indices, j]
                    print("child_scores[{}, {}]:".format(i, j),
                          child_scores_ij)
                    var_sfx = "{}_{}".format(i, j)
                    copy_indices, probs2copy, alpha_indices, alpha = \
                        self._forward_node(
                            var_sfx, prnt_scores_i,
                            child_scores_ij, rels_i[inst_indices, j]
                        )
                    print("copy_indices", copy_indices)
                    print("probs2copy", probs2copy)
                    print("alpha_indices", alpha_indices)
                    print("alpha", alpha)
        return self.M


a = AlphaModel(6, 3)

node_scores = tensor(softmax(np.random.uniform(0, 1, (2, 6, 3)), -1))
node_scores[0][0] = 0.
node_scores[0][1] = 0.
node_scores[1][0] = 0.
print("node_scores", node_scores)

children = np.zeros(node_scores.shape, dtype=np.int32)
children[0][-1] = [2, 3, 4]
children[1, [2, 3, 4, 5], 0] = [1, 2, 3, 4]
print("children", children)

rels = np.zeros(node_scores.shape, dtype=np.int32)
rels[0][-1] = [1, 2, 3]
rels[1, [2, 3, 4, 5], 0] = 2
print("rels", rels)

labels = np.array([0, 1], dtype=np.int32)
print("labels", labels)
print("a():", a(node_scores, children, rels, labels))
