#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Collection of variables and constants common to all modules.

Attributes:

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from functools import wraps
from datetime import datetime
import logging


##################################################################
# Variables and Constants
LOG_LVL = logging.INFO
LOGGER = logging.getLogger("DASA")
LOGGER.setLevel(LOG_LVL)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setLevel(LOG_LVL)
sh.setFormatter(formatter)
LOGGER.addHandler(sh)


##################################################################
# Decorators
class timeit(object):
    """Decorator class for measuring time performance.

    Attributes:
      msg (str): message to be printed on method invocation

    """

    def __init__(self, a_msg):
        """Class constructor.

        Args:
          a_msg (str): debug message to print

        """
        self.msg = a_msg

    def __call__(self, a_func):
        """Decorator function.

        Args:
          a_func (method): decorated method

        Returns:
          method: wrapped method

        """
        def _wrapper(*args, **kwargs):
            LOGGER.info(self.msg + " started")
            start_time = datetime.utcnow()
            a_func(*args, **kwargs)
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).total_seconds()
            LOGGER.info(self.msg + " finished (%2f sec)",
                        time_delta)
        return wraps(a_func)(_wrapper)
