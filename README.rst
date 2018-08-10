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

  pwd
  /home/sidorenko/Projects/RSTParser

  git rev-parse HEAD
  8b595c3913daa68745758c1eb3420bfa90cbb264

  for f in ../DASA/data/\*/\*/\*.json; do \
    ./scripts/add_rst_trees bhatia data/pcc-dis-bhatia/test/rstparser.bhatia.model $f > 1 && \
    mv 1 $f;
  done


Examples
--------

Last EDU
^^^^^^^^

To predict the polarity of a tweet based on the polarity of the last
EDU, we used the following command to create the model:

.. code-block:: shell

  dasa_sentiment -v train -t last data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then the following scripts to predict the label and evaluate the
quality:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/last/last.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/last/last.json

equivalently:

.. code-block:: shell
  dasa_sentiment -v train -t last data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/last/last.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/last/last.json


Results
~~~~~~~

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.52      0.83      0.64       437
   negative       0.57      0.17      0.26       209
   neutral       0.61      0.43      0.50       360
   avg / total       0.57      0.55      0.51      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 45.30%
   Micro-Averaged F1-Score (All Classes): 54.8708%

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.56      0.55      0.56       190
   negative       0.46      0.29      0.36       113
   neutral       0.73      0.80      0.76       447
   avg / total       0.65      0.66      0.65       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 45.86%
   Micro-Averaged F1-Score (All Classes): 66.1333%

+-------+---------------------------+---------------------------+---------------------------+-------------------+-------------------+
| Data  |          Positive         |           Negative        |          Neutral          | :math:`Macro F_1` | :math:`Micro F_1` |
+       +------+------+-------------+------+------+-------------+------+------+-------------+                   +                   +
|       |   P  |   R  | :math:`F_1` |   P  |   R  | :math:`F_1` |   P  |   R  | :math:`F_1` |                   |                   |
+-------+------+------+-------------+------+------+-------------+------+------+-------------+-------------------+-------------------+
| PotTS | 0.52 | 0.83 |     0.64    | 0.57 | 0.17 |     0.26    | 0.61 | 0.43 |      0.5    |        0.453      |        0.549      |
| SB10k | 0.56 | 0.55 |     0.56    | 0.46 | 0.29 |     0.36    | 0.73 | 0.8  |     0.76    |        0.459      |        0.661      |
+-------+------+------+-------------+------+------+-------------+------+------+-------------+-------------------+-------------------+

