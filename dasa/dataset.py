#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for accessing training and development data.

Attributes:
  Dataset (class): class for accessing training and development data

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from torch.utils.data import TensorDataset as TDS
import torch

from .constants import DFLT_DTYPE


##################################################################
# Class
class Dataset(TDS):
    def __init__(self, x, y):
        """Class constructor.

        Args:
          x (np.array): digitized input data
          y (np.array): digitized labels

        """
        super(Dataset, self).__init__(
            torch.from_numpy(x).to(DFLT_DTYPE),
            torch.from_numpy(y)
        )
