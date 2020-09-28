#!/usr/bin/env python

##################################################################
# Imports
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from scipy.special import softmax
from torch import tensor
from pyro.distributions import (Beta, MultivariateNormal)
from torch.nn import Softmax
import numpy as np
import pyro
import torch


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
        # print("Mu", Mu, Mu.shape)
        Mu *= np.arange(self._n_rels).reshape(self._n_rels, 1, 1, 1)
        # print("*Mu", Mu, Mu.shape)
        return tensor(Mu)

    @PyroParam
    def M_Sigma(self):
        Sigma = np.eye(self._n_polarities, dtype="float32")
        # print("Sigma", Sigma, Sigma.shape)
        return tensor(Sigma)

    @PyroSample
    def M(self):
        M = MultivariateNormal(self.M_Mu, self.M_Sigma)
        # print("M", M, list(M.batch_shape))
        # print("M.batch_shape", M.batch_shape)
        # print("M.event_shape", M.event_shape)
        return M

    @PyroParam
    def beta_p(self):
        return 5. * tensor(np.ones((self._n_rels, self._n_polarities, 1),
                                   dtype="float32"))

    @PyroParam
    def beta_q(self):
        return 5. * tensor(np.ones((self._n_rels, self._n_polarities, 1),
                                   dtype="float32"))

    @PyroSample
    def beta(self):
        beta = Beta(self.beta_p, self.beta_q)
        print("beta", beta, list(beta.batch_shape))
        print("beta.batch_shape", beta.batch_shape)
        print("beta.event_shape", beta.event_shape)
        return beta

    def _get_child_scores(self, node_scores, children, inst_indices, i,
                          n_instances, max_children):
        child_indices = children[inst_indices, i].reshape(-1)
        # print("child_indices:", child_indices)
        child_scores = node_scores[
            inst_indices.repeat(max_children, 1).t().reshape(-1),
            child_indices
        ].reshape(n_instances, max_children, -1)
        # print("* child_indices:", child_indices)
        return child_scores

    def _forward_node(self, var_sfx: str, prnt_probs: tensor,
                      child_probs, rels):
        # The vector `nz_chld_indices` will contain indices of the child_probs
        # which are not zero.
        nz_chld_indices = child_probs.sum(dim=-1).nonzero().squeeze(-1)
        # print("nz_chld_indices:", nz_chld_indices)
        if nz_chld_indices.nelement() == 0:
            return None, None, None, None
        # Only leave `parent`, `child`, and `rels` elements for which
        # `child_probs` are not zero.
        child_probs = child_probs[nz_chld_indices].unsqueeze(1)
        # print("* child_probs:", child_probs)
        rels = rels[nz_chld_indices]
        # print("nz_rels:", rels)
        # print("M:", self.M, self.M.shape)
        M_nz_rels = self.M[rels, :, nz_chld_indices, :]
        # print("M[nz_rels]:", M_nz_rels, M_nz_rels.shape)
        M_nz_rels = M_nz_rels.reshape(
            -1, self._n_polarities, self._n_polarities
        )
        # print("* M[nz_rels]:", M_nz_rels, M_nz_rels.shape)
        # print("* child_probs:", child_probs, child_probs.shape)
        child_probs = torch.bmm(child_probs, M_nz_rels)
        # print("++ child_probs:", child_probs)
        child_probs = self._softmax(child_probs)
        # print("^^ child_probs:", child_probs)
        prnt_probs = prnt_probs[nz_chld_indices]
        prnt_probs_sum = prnt_probs.sum(dim=-1)
        z_prnt_indices = (prnt_probs_sum == 0.).nonzero().squeeze_(-1)
        # print("z_prnt_indices:", z_prnt_indices)
        # indices of instances whose child scores are non-zero, but parent
        # scores are zero
        copy_indices = nz_chld_indices[z_prnt_indices]
        child_probs2copy = child_probs[z_prnt_indices]
        nz_prnt_indices = prnt_probs_sum.nonzero().squeeze(-1)
        # print("rels:", rels)
        # print("nz_prnt_indices:", nz_prnt_indices)
        alpha_indices = nz_chld_indices[nz_prnt_indices]
        # print("alpha_indices:", alpha_indices)
        if alpha_indices.nelement() == 0:
            alpha = None
        else:
            child_probs = child_probs[nz_prnt_indices]
            prnt_probs = prnt_probs[nz_prnt_indices]
            rels = rels[nz_prnt_indices]
            # take a convex combination of the parent and child scores as new
            # values for alpha and scale the resulting alpha scores with the
            # corresponding scale factor.
            print("self.beta:", self.beta)
            print("rels:", rels)
            print("self.beta[rels]:", self.beta[rels, :, nz_prnt_indices])
            beta = self.beta[rels, ]
            alpha = (1. - beta) * prnt_probs + beta * child_probs
            scale = self.scale(prnt_probs, child_probs).unsqueeze_(-1)
            alpha *= scale
        return copy_indices, child_probs2copy, alpha_indices, alpha

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
                print("rels[{}]:".format(i), rels_i)
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

node_scores = tensor(
    softmax(
        np.random.uniform(0, 1, (2, 6, 3)), -1)
).to(torch.float32)
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
rels = tensor(rels, dtype=torch.long)
print("rels", rels)

labels = np.array([0, 1], dtype=np.int32)
print("labels", labels)
print("a():", a(node_scores, children, rels, labels))
