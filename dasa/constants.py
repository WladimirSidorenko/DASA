#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Collection of variables and constants common to all modules.

Attributes:

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from six import iteritems
import torch
import os

##################################################################
# Constants
DFLT_ENCODING = "utf-8"
DFLT_DTYPE = torch.get_default_dtype()

DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
DFLT_MODEL_PATH = os.path.join(MODEL_DIR, "dasa.model")

# Methods
LAST = "last"
ROOT = "root"
R2N2 = "r2n2"
DDR = "ddr"
RDM = "rdm"
WANG = "wang"
NO_DISCOURSE = "no-discourse"

# Relation Schemes
BHATIA = "bhatia"
CHENLO = "chenlo"
GOLD = "gold"
HEERSCHOP = "heerschop"
PCC = "pcc"
ZHOU = "zhou"

# Sentiment Scores
SOCAL = "socal"
XLNET = "xlnet"

# mapping from polarity classes to indices and vice versa
POSITIVE = "positive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
NEUTRAL = "neutral"
MIXED = "mixed"
CLS2IDX = {NEGATIVE: 0, NEUTRAL: 1, POSITIVE: 2}
IDX2CLS = {v: k for k, v in iteritems(CLS2IDX)}
N_POLARITIES = len(CLS2IDX)

BUFFER_FACTOR = 1.4

CONTRASTIVE_RELS = {
    "antithesis", "antithesis-e", "concession", "otherwise",
    "contrast", "comparison", "consequence-s", "problem-solution",
    "contrastive"
}
