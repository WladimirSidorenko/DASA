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

This package provides several implementations of common
discourse-aware sentiment analysis (DASA) methods.  Most of these
approaches infer the overall polarity of input document (*e.g,* of a
tweet) from the polarity scores of its elementary discourse units
(EDUs) by either accumulating these scores over the RST tree or
choosing a single EDU, which is most representative of the whole
analyzed text (*e.g.*, the last or root discourse segment).

Data Preparation
----------------

We use the PotTS_ and SB10k_ corpora as primary data for evaluation.

Tagging, Parsing, and Discourse Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before using these corpora, we have processed all tweets of these
datasets with the `text normalization pipeline`_ of Sidarenka et al.,
and parsed them using the `Mate dependency parser`_ of Bohnet et al.
Afterwards, we have converted the resulting CoNLL files into the TSV
format using the scipt ``conll2tsv_``, and subsequently exported the
resulting TSV into JSON with the help of the script ``tsv2json_``.  In
addition to that, we have also added information about discourse
segments and automatically predicted sentiment scores for each of
these segements with the scripts `scripts/add_segmentation` and
`scripts/add_polarity_scores` respectively.

Discourse Parsing
^^^^^^^^^^^^^^^^^

To derive RST trees for the obtained tweets, we used the script
`add_rst_trees` from the `RSTParser package`_:

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

To predict the polarity of a tweet based on the polarity of its last
EDU, we used the following command to create the model:

.. code-block:: shell

  dasa_sentiment -v train -t last data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then executed the following scripts to predict the label and
evaluate the quality:

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

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.52  | 0.83 |      0.64     |  0.57  | 0.17 |     0.26      |  0.61  | 0.43 |     0.5     |       0.453       |       0.549       |
| SB10k     |  0.56  | 0.55 |      0.56     |  0.46  | 0.29 |     0.36      |  0.73  | 0.8  |     0.76    |       0.459       |       0.661       |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+


Root EDU
^^^^^^^^

To predict the polarity of a tweet based on the root EDU (*i.e.*, the
nucleus of the nucleus), we used the following commands to create and
test the models:

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
   positive      0.56      0.73      0.64       437
   negative      0.58      0.22      0.32       209
   neutral       0.55      0.54      0.54       360
   avg / total       0.56      0.56      0.54      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 48.12%
   Micro-Averaged F1-Score (All Classes): 55.9642%

.. comment:

   General Statistics:
   precision    recall  f1-score   support
   positive      0.51      0.55      0.53       190
   negative      0.40      0.30      0.35       113
   neutral       0.74      0.76      0.75       447
   avg / total       0.63      0.64      0.63       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 43.77%
   Micro-Averaged F1-Score (All Classes): 64.0000%


+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.56  | 0.73 |      0.64     |  0.58  | 0.22 |     0.32      |  0.55  | 0.54 |    0.54     |       0.481       |       0.5596      |
| SB10k     |  0.51  | 0.55 |      0.53     |  0.4   | 0.3  |     0.35      |  0.74  | 0.76 |    0.75     |       0.438       |       0.64        |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+


.. _PotTS: http://www.lrec-conf.org/proceedings/lrec2016/pdf/97_Paper.pdf
.. _SB10k: http://aclweb.org/anthology/W17-1106
.. _text normalization pipeline: https://www-archiv.tu-darmstadt.de/gscl2013/images/sidarenka_scheffler_stede.pdf
.. _Mate dependency parser: http://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/matetools.en.html
.. _conll2tsv: https://github.com/WladimirSidorenko/CGSA/blob/master/scripts/conll2tsv
.. _tsv2json: https://github.com/WladimirSidorenko/DASA/blob/master/scripts/tsv2json
.. _add_segmentation: https://github.com/WladimirSidorenko/DASA/blob/master/scripts/add_segmentation
.. _add_polarity_scores: https://github.com/WladimirSidorenko/DASA/blob/master/scripts/add_polarity_scores
.. _RSTParser package: https://github.com/WladimirSidorenko/RSTParser
