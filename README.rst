Discourse-Aware Sentiment Analysis
==================================
.. image:: https://travis-ci.org/WladimirSidorenko/DASA.svg?branch=master
   :alt: Build Status
   :target: https://travis-ci.org/WladimirSidorenko/DASA

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :alt: MIT License
   :align: right
   :target: http://opensource.org/licenses/MIT

*This project is currently work in progress.*


Description
-----------

This package provides several approaches to discourse-aware sentiment
analysis (DASA).  Most of these methods infer the overall sentiment
class of the document from the polarity scores of its elementary
discourse units (EDUs).

Data Preparation
----------------

Tagging, Parsing, and Discourse Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data used in project are taken from the PotTS and SB10k corpora.
After processing these tweets with the `text normalization` pipeline
`Mate` Dependency Parser, we have converted the resulting CoNLL files
into the TSV format using the script `scripts/tsv2json`, and added
information about discourse segments and automatically predicted
sentiment scores with the scripts `scripts/add_segmentation` and
`scripts/add_polarity_scores` respectively.

Discourse Parsing
^^^^^^^^^^^^^^^^^

To derive RST trees for the obtained tweets, we have used the script
`add_rst_trees` from the [RSTParser package]():

.. code-block:: shell

  >pwd
  /home/sidorenko/Projects/RSTParser

  >git rev-parse HEAD
  8b595c3913daa68745758c1eb3420bfa90cbb264

  >for f in ../DASA/data/\*/\*/\*.json; do \
     ./scripts/add_rst_trees bhatia data/pcc-dis-bhatia/test/rstparser.bhatia.model $f > 1 && \
     mv 1 $f;
   done
