#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

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


##################################################################
# Class
class Dataset(TDS):
    def __init__(self, *x):
        """Class constructor.

        Args:
          x (list[np.array]): digitized input data

        """
        x = [torch.from_numpy(x_) for x_ in x[:-1]]
        y = torch.from_numpy(x[-1])
        x.append(y)
        super(Dataset, self).__init__(*x)
