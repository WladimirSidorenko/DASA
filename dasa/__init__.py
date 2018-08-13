#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Main package for discourse-awar sentiment analysis.

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from .base import DASBaseAnalyzer
from .last import DASLastAnalyzer
from .no_discourse import DUSAnalyzer
from .root import DASRootAnalyzer


##################################################################
# Variables and Constants
__name__ = "dasa"
__all__ = ["DASBaseAnalyzer", "DASLastAnalyzer", "DASRootAnalyzer",
           "DUSAnalyzer"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.1.0a0"
