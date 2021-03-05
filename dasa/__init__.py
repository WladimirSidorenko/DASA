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
from .ddr import DDRAnalyzer
from .last import LastAnalyzer
from .no_discourse import DUSAnalyzer
from .root import RootAnalyzer
from .r2n2 import R2N2Analyzer
from .rdm import RDAnalyzer
from .wang import WangAnalyzer


##################################################################
# Variables and Constants
__name__ = "dasa"
__all__ = ["DASBaseAnalyzer", "DDRAnalyzer", "DUSAnalyzer", "LastAnalyzer",
           "RootAnalyzer", "R2N2Analyzer", "RDAnalyzer", "WangAnalyzer"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.1.0a0"
