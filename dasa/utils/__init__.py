#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Package containing common utility functions.

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from collections import defaultdict
from random import sample, shuffle
from six import itervalues

from .common import LOGGER


##################################################################
# Methods
def balance_data(x, y):
    """Create balanced dataset with equal number of instances for all classes.

    Args:
      x (list): input instances
      y (list): labels


    Returns:
      2-tuple: balanced versions of `x' and `y'

    """
    label2idx = defaultdict(list)
    for i, y_i in enumerate(y):
        label2idx[y_i].append(i)
    max_instances = min([len(v) for v in itervalues(label2idx)])
    balanced_indices = []
    for y_indices in itervalues(label2idx):
        balanced_indices.extend(sample(y_indices, max_instances))
    shuffle(balanced_indices)
    balanced_x = [x[i] for i in balanced_indices]
    balanced_y = [y[i] for i in balanced_indices]
    return (balanced_x, balanced_y)


##################################################################
# Variables and Constants
__all__ = ["LOGGER", "balance_data"]
__name__ = "cgsa.utils"
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.0"
