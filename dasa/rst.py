#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

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

import numpy as np


##################################################################
# Classes
class Tree(object):
    def __init__(self, data, root):
        """Class constructor.

        Args:
          data (dict): data pertaining to the discourse tree
          root (dict): attributes of the root node

        """
        self._id = int(root["id"])
        self._is_leaf = self._id >= 0
        self._rel2par = root["rel2par"]
        self._ns = root["n/s"]
        self._root_edus = None
        self._children = [Tree(data, ch) for ch in root["children"]]
        self.toks = []
        self.polarity_scores = []
        if self._is_leaf:
            edu = data["edus"][self._id]
            toks = data["toks"]
            self.toks = [toks[t] for t in edu["toks"]]
            self.polarity_scores = np.array(edu["polarity_scores"])

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
                                   if ch._ns == "Nucleus"]
        return self._root_edus

    def __repr__(self):
        tree = ("<RSTTree id={:d} leaf={:d} rel2par={:s} n/s={:s}"
                " children=[{:s}]>").format(
                    self._id, self._is_leaf, self._rel2par, self._ns,
                    ' '.join([repr(ch) for ch in self._children]))
        return tree
