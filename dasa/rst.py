#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Module providing a class for accessing and analyzing RST trees.

Attributes:
  Node (class): discourse node, whcih serves as a component of RST tree
  Tree (class): RST discourse tree

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from itertools import chain
import numpy as np

from .constants import IDX2CLS


##################################################################
# Classes
class Tree(object):
    @staticmethod
    def _find_nuc_prnt(self, node, heads, prnt=None):
        """Find parent of nucleus EDU.

        Args:
          node (Tree):

        """
        if False:
            pass

    def __init__(self, data, root, prnt=None):
        """Class constructor.

        Args:
          data (dict): data pertaining to the discourse tree
          root (dict): attributes of the root node

        """
        self._id = int(root["id"])
        self._is_leaf = self._id >= 0
        self._depth = -1
        self._len = -1
        self._width = -1
        self._rel2par = root["rel2par"]
        self._ns = root["n/s"]
        self._prnt = prnt
        self._children = [Tree(data, ch, self) for ch in root["children"]]
        self.toks = []
        self._root_edus = None
        self.polarity_scores = []
        if self._is_leaf:
            self._leaves = [self]
            edu = data["edus"][self._id]
            toks = data["toks"]
            self.toks = [toks[t] for t in edu["toks"]]
            self.polarity_scores = np.array(edu["polarity_scores"])
        else:
            self._leaves = [leaf
                            for ch in self._children
                            for leaf in ch.leaves]

    def __len__(self):
        """Total number of nodes in th

        """
        if self._len < 0:
            self._len = sum(len(ch) for ch in self._children) + 1
        return self._len

    def bfs(self):
        """Iterate over all tree nodes.

        Yields:
          all tree nodes in bfs order

        """
        nodes = [self]
        visited_nodes = set([self._id])
        while nodes:
            yield nodes
            nodes = [ch
                     for node_i in nodes
                     for ch in node_i._children
                     if ch._id not in visited_nodes]
            for node in nodes:
                visited_nodes.add(node._id)

    def dfs(self):
        """Iterate over all tree nodes.

        Yields:
          all tree nodes in dfs order

        """
        yield self
        for child in self.children:
            for grandchild in iter(child):
                yield grandchild

    @property
    def children(self):
        return self._children

    @property
    def depth(self):
        if self._depth < 0:
            if len(self._children) == 0:
                self._depth = 1
            else:
                self._depth = 1 + max([ch.depth for ch in self.children])
        return self._depth

    @property
    def id(self):
        return self._id

    @property
    def leaves(self):
        return self._leaves

    @property
    def ns(self):
        return self._ns

    @property
    def parent(self):
        return self._prnt

    @property
    def rel2par(self):
        return self._rel2par

    @rel2par.setter
    def rel2par(self, value):
        self._rel2par = value

    @property
    def width(self):
        if self._width < 0:
            self._width = len(self.children)
        return self._width

    @property
    def root_edus(self):
        """Root EDU(s) of the given RST tree.

        Get a list of root edus which are the nuclei of the top-most nuclei.

        Returns:
          list: root EDU(s)

        """
        if self._root_edus is None:
            if self._is_leaf:
                self._root_edus = [self]
            else:
                self._root_edus = [edu
                                   for ch in self._children
                                   for edu in ch.root_edus
                                   if ch.ns == "Nucleus"]
        return self._root_edus

    def to_deps(self):
        """Convert given rst tree to discourse dependency represenation.

        See Hirao et al. (2013) for particular details about this conversion.

        """
        nuclei = [leaf for leaf in self.leaves if leaf.ns == "Nucleus"]
        # compute heads
        heads = self._compute_heads(nuclei)
        # convert nucleus and satellite EDUs to dg nodes
        satellites = [leaf for leaf in self.leaves if leaf.ns == "Satellite"]
        id2dg_node = {leaf._id: DepDT(leaf._id, leaf.polarity_scores)
                      for leaf in chain(nuclei, satellites)}
        dg_root = self._nuclei2dg_tree(nuclei, id2dg_node, heads)
        self._satellites2dg_tree(satellites, id2dg_node, heads)
        return dg_root

    def _compute_heads(self, nuclei):
        """Compute heads of RST nodes.

        Args:
          nuclei (list[Tree]):

        Returns:
          dict: mapping from node ids to

        """
        heads = {nuc._id: nuc for nuc in nuclei}
        nodes = nuclei[:]
        while nodes:
            node = nodes.pop()
            node_id = node._id
            head = heads[node_id]
            prnt = node.parent
            # skip satellites and the root node
            if node.ns == "Satellite" or prnt is None:
                continue
            prnt_id = prnt._id
            head_id = head._id
            if prnt_id in heads:
                if head_id < heads[prnt_id]._id:
                    heads[prnt_id] = head
            else:
                heads[prnt_id] = head
                nodes.append(prnt)
        return heads

    def _nuclei2dg_tree(self, nuclei, id2dg_node, heads):
        """Contruct dg tree from nucleus EDUs.

        Args:
          nuclei (list[Tree]): EDUs to construct the tree from
          id2dg_node (dict): mapping from EDU ids to DG nodes
          heads (dict): mapping from EDU ids to DG nodes

        Returns:
          dict: mapping from node ids to

        """
        dg_root = DepDT(-1, [0.] * len(IDX2CLS))
        for nuc_edu in nuclei:
            edu_id = nuc_edu._id
            dg_node = id2dg_node[edu_id]
            dg_prnt = None
            prnt = nuc_edu.parent
            rel2par = nuc_edu._rel2par
            # find closest satellite
            while prnt is not None:
                if rel2par is None or rel2par == "span":
                    rel2par = prnt._rel2par
                if prnt.ns != "Nucleus":
                    break
                prnt = prnt.parent
            if prnt.parent is None:
                dg_root.add_child(dg_node, rel2par)
            else:
                prnt = prnt.parent
                head = heads[prnt._id]
                dg_prnt = id2dg_node[head._id]
                dg_prnt.add_child(dg_node, rel2par)
        return dg_root

    def _satellites2dg_tree(self, satellites, id2dg_node, heads):
        """Contruct dg tree from nucleus EDUs.

        Args:
          satellites (list[Tree]): EDUs to add to the DG tree
          id2dg_node (dict): mapping from EDU ids to DG nodes
          heads (dict): mapping from EDU ids to their heads

        Returns:
          None:

        """
        for sat_edu in satellites:
            dg_node = id2dg_node[sat_edu._id]
            rel2par = sat_edu._rel2par
            prnt = sat_edu.parent
            while prnt is not None and prnt._is_leaf:
                prnt = prnt.parent
            assert prnt is not None, \
                "Satellite node {:s} has no parent.".format(sat_edu._id)
            head = heads[prnt._id]
            head_node = id2dg_node[head._id]
            head_node.add_child(dg_node, rel2par)

    def __repr__(self):
        tree = ("<RSTTree id={} leaf={} rel2par={} n/s={}"
                " children=[{:s}]>").format(
                    self._id, self._is_leaf, self._rel2par, self.ns,
                    ", ".join([repr(ch) for ch in self._children])
                )
        return tree


class DepDT(object):
    """Dependency-based discourse tree (Hirao et al., 2013).

    """

    def __init__(self, _id, polarity_scores):
        """Class constructor.

        Args:
          _id (int): id of the given node.
        """
        self._id = _id
        self._prnt = None
        self._rel2par = None
        self._children = []
        self.polarity_scores = polarity_scores

    def add_child(self, dg_node, rel2par=None):
        """Add child to the given node.

        Args:
          dg_node (DepDT): dependency node
          rel2prnt (str): link connecting the child to the given node

        """
        if rel2par and rel2par.lower() == "span":
            rel2par = None
        self._children.append(dg_node)
        dg_node._prnt = self
        dg_node._rel2par = rel2par

    def bfs(self):
        """Traverse graph in breadth-first search order.

        Args:

        Yields:
          [list]: list of nodes

        """
        nodes = [self]
        visited_nodes = set([self._id])
        while nodes:
            yield nodes
            nodes = [ch
                     for node_i in nodes
                     for ch in node_i._children
                     if ch._id not in visited_nodes]
            for node in nodes:
                visited_nodes.add(node._id)

    def to_dotty(self, header=False):
        """Return dotty representation of the given graph.

        Args:
          header (bool): add graph heade

        Returns:
          str: dotty representation of the given graph

        """
        ret = "{:d} [label=\"{:d}\"];\n".format(self._id, self._id)
        for ch in self._children:
            ret += "{:d} -- {:d} [label=\"{}\"];\n".format(
                self._id, ch._id, ch._rel2par)
            ret += ch.to_dotty(False)
        if header:
            ret = """\
graph {
graph [fontsize=24, forcelabels=true, fontname=fixed];
edge [fontsize=24, labelfontsize=20, fontname=fixed];
node [fontsize=24, labelfontsize=20, fontname=fixed,
style=filled, fillcolor=lightskyblue];""" + ret + "}"
        return ret

    def __repr__(self):
        tree = ("<DepDT id={} rel2par={} polarity_score={!r}"
                " children=[{}]>").format(
                    self._id, self._rel2par, self.polarity_scores,
                    ", ".join([repr(ch) for ch in self._children])
                )
        return tree
