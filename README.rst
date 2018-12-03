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
approaches infer the overall polarity of the input (*e.g,* of a tweet)
from the polarity scores of its elementary discourse units (EDUs) by
either accumulating these scores over the RST tree or choosing a
single EDU, which is most representative of the whole analyzed text
(*e.g.*, the last discourse segment).

Data Preparation
----------------

We use PotTS_ and SB10k_ as primary data sources for evaluation.

Tagging, Parsing, and Discourse Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before using these corpora, we processed all tweets of these datasets
with the `text normalization pipeline`_ [SIDARENKA]_ and parsed them
using the `Mate dependency parser`_ [BOHNET]_.  Afterwards, we
converted the resulting CoNLL files into the TSV format using the
scipt conll2tsv_, and subsequently exported the resulting TSV into
JSON with the script tsv2json_.  In addition to that, we also added
information about discourse segments and automatically predicted
sentiment scores for each of these segements with the scripts
`add_segmentation`_ and `add_polarity_scores`_ respectively.

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

DDR
^^^

To determine the polarity of a tweet using the discourse depth
reweighting (DDR) method [BHATIA]_, you can use the following command to
create the model:

.. code-block:: shell

  dasa_sentiment -v train -t ddr -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then execute the following scripts to predict the labels for the
test sets and evaluate the quality of the resulting model:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/root/root.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/root/root.json

Equivalently, you can run the following commands to check the
performance of this approach on the SB10k_ corpus:

.. code-block:: shell

  dasa_sentiment -v train -t ddr -r bhatia data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/ddr/ddr.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/ddr/ddr.json


Results
~~~~~~~

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.73      0.77      0.75       437
   negative       0.54      0.59      0.56       209
   neutral       0.69      0.61      0.65       360
   avg / total       0.68      0.67      0.67      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 65.50%
   Micro-Averaged F1-Score (All Classes): 67.3956%

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.59      0.63      0.61       190
   negative       0.48      0.44      0.46       113
   neutral       0.77      0.76      0.77       447
   avg / total       0.68      0.68      0.68       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 53.39%
   Micro-Averaged F1-Score (All Classes): 68.1333%

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.73  | 0.77 |      0.75     |  0.54  | 0.59 |     0.56      |  0.69  | 0.61 |     0.65    |       0.655       |      0.674        |
| SB10k     |  0.59  | 0.63 |      0.61     |  0.48  | 0.44 |     0.46      |  0.77  | 0.76 |     0.77    |       0.534       |      0.681        |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

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


No-Discourse
^^^^^^^^^^^^

To predict the polarity of a tweet discregarding the discourse
information, you can invoke the above scripts as follows:

.. code-block:: shell

  dasa_sentiment -v train -t no-discourse data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then the following scripts to predict the label and evaluate the
quality:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/no-discourse/no-discourse.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/no-discourse/no-discourse.json

equivalently:

.. code-block:: shell

  dasa_sentiment -v train -t no-discourse data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/no-discourse/no-discourse.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/no-discourse/no-discourse .json


Results
~~~~~~~

.. comment:

   General Statistics:
   precision    recall  f1-score   support
   positive       0.73      0.82      0.77       437
   negative       0.61      0.56      0.58       209
   neutral       0.72      0.66      0.69       360
   avg / total       0.70      0.71      0.70      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 67.66%
   Micro-Averaged F1-Score (All Classes): 70.5765%

.. comment:

   General Statistics:
   precision    recall  f1-score   support
   positive       0.64      0.69      0.66       190
   negative       0.45      0.45      0.45       113
   neutral       0.82      0.79      0.80       447
   avg / total       0.72      0.71      0.71       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 55.72%
   Micro-Averaged F1-Score (All Classes): 71.3333%


+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.73  | 0.82 |      0.77     |  0.61  | 0.56 |     0.58      |  0.72  | 0.66 |    0.69     |       0.677       |       0.706       |
| SB10k     |  0.64  | 0.69 |      0.66     |  0.45  | 0.45 |     0.45      |  0.82  | 0.79 |    0.8      |       0.557       |       0.713       |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

Root EDU
^^^^^^^^

To predict the polarity of a tweet based on the root EDU (*i.e.*, the
nucleus of the nucleus), we used the following commands to create and
test the models:

.. code-block:: shell

  dasa_sentiment -v train -t root -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then the following scripts to predict the label and evaluate the
quality:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/root/root.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/root/root.json

equivalently:

.. code-block:: shell

  dasa_sentiment -v train -t root -r bhatia data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/root/root.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/root/root.json


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

R2N2
^^^^

To determine the polarity of a tweet using rhetorical recursive neural
networks (R2N2) [BHATIA]_, you can use the following command to create
the model:

.. code-block:: shell

  dasa_sentiment -v train -t r2n2 -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then run:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/r2n2/r2n2.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/r2n2/r2n2.json

to predict the labels on the test sets and evaluate the quality of the
resulting model.

Equivalently, you can run the following commands to check the
performance of this approach on the SB10k_ corpus:

.. code-block:: shell

  dasa_sentiment -v train -t r2n2 -r bhatia data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/r2n2/r2n2.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/r2n2/r2n2.json


Results
~~~~~~~

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.74      0.78      0.76       437
   negative       0.59      0.53      0.56       209
   neutral       0.68      0.68      0.68       360
   avg / total       0.69      0.69      0.69      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 65.72%
   Micro-Averaged F1-Score (All Classes): 69.1849%

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.64      0.69      0.66       190
   negative       0.46      0.45      0.45       113
   neutral       0.81      0.79      0.80       447
   avg / total       0.72      0.71      0.71       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 55.92%
   Micro-Averaged F1-Score (All Classes): 71.3333%

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.74  | 0.78 |      0.76     |  0.59  | 0.53 |     0.56      |  0.68  | 0.68 |     0.68    |       0.6572      |      0.6918       |
| SB10k     |  0.64  | 0.69 |      0.66     |  0.46  | 0.45 |     0.45      |  0.81  | 0.79 |     0.8     |       0.5592      |      0.7133       |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

RDP
^^^

To determine the polarity of a tweet using a recursive Dirichlet
process (RDP), you can use the following command to train the model:

.. code-block:: shell

  dasa_sentiment -v train -t rdp -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and then run:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/rdp/rdp.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/rdp/rdp.json

to predict the labels on the test sets and evaluate the quality of the
resulting model.

Equivalently, you can run the following commands to check the
performance of this approach on the SB10k_ corpus:

.. code-block:: shell

  dasa_sentiment -v train -t rdp -r bhatia data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/rdp/rdp.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/rdp/rdp.json


Results
~~~~~~~

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.71      0.83      0.77       437
   negative       0.60      0.55      0.58       209
   neutral       0.73      0.62      0.67       360
   avg / total       0.70      0.70      0.69      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 67.03%
   Micro-Averaged F1-Score (All Classes): 69.6819%

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.64      0.69      0.67       190
   negative       0.45      0.45      0.45       113
   neutral       0.82      0.79      0.80       447
   avg / total       0.72      0.71      0.72       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 55.88%
   Micro-Averaged F1-Score (All Classes): 71.4667%


+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.71  | 0.83 |      0.77     |  0.59  | 0.55 |     0.57      |  0.73  | 0.61 |     0.66    |       0.667       |      0.693        |
| SB10k     |  0.64  | 0.69 |      0.67     |  0.45  | 0.45 |     0.45      |  0.82  | 0.79 |     0.8     |       0.559       |      0.715        |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

WANG
^^^^

To determine the polarity of a message using a linear combination of
EDU polarities [WANG]_, you can use the following command to create
the model:

.. code-block:: shell

  dasa_sentiment -v train -t wang -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json

and run:

.. code-block:: shell

  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/wang/wang.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/wang/wang.json

to predict the labels on the test sets and evaluate the quality of the
resulting model.


Results
~~~~~~~

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.58      0.79      0.67       437
   negative       0.61      0.21      0.31       209
   neutral       0.61      0.57      0.59       360
   avg / total       0.59      0.59      0.57      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 48.72%
   Micro-Averaged F1-Score (All Classes): 59.0457%

.. comment:
   General Statistics:
   precision    recall  f1-score   support
   positive       0.61      0.63      0.62       190
   negative       0.46      0.29      0.36       113
   neutral       0.76      0.82      0.79       447
   avg / total       0.68      0.69      0.68       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 48.84%
   Micro-Averaged F1-Score (All Classes): 69.3333%

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.58  | 0.79 |      0.67     |  0.61  | 0.21 |     0.31      |  0.61  | 0.57 |     0.59    |       0.4872      |      0.5905       |
| SB10k     |  0.61  | 0.63 |      0.62     |  0.46  | 0.29 |     0.36      |  0.76  | 0.82 |     0.79    |       0.4884      |      0.6933       |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

LCRF
^^^^

In the same way, you can use the ``-t lcrf`` option, to train and
evaluate latent CRFs:

.. code-block:: shell

  dasa_sentiment -v train -t lcrf -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json
  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/lcrf/lcrf.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/lcrf/lcrf.json


Results
~~~~~~~

.. comment: PotTS
   precision    recall  f1-score   support
   positive       0.76      0.79      0.77       437
   negative       0.61      0.53      0.57       209
   neutral       0.70      0.71      0.70       360
   avg / total       0.70      0.71      0.71      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 67.02%
   Micro-Averaged F1-Score (All Classes): 70.7753%

.. comment: SB10k
   General Statistics:
   precision    recall  f1-score   support
   positive       0.64      0.69      0.66       190
   negative       0.45      0.45      0.45       113
   neutral       0.82      0.79      0.80       447
   avg / total       0.72      0.71      0.71       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 55.72%
   Micro-Averaged F1-Score (All Classes): 71.3333%

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.76  | 0.79 |      0.77     |  0.61  | 0.53 |     0.57      |  0.7   | 0.71 |     0.7     |       0.67        |      0.708        |
| SB10k     |  0.64  | 0.69 |      0.66     |  0.45  | 0.45 |     0.45      |  0.82  | 0.79 |     0.8     |       0.557       |      0.713        |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

LMCRF
^^^^^

In the same way, you can use the ``-t lmcrf`` option, to train and
evaluate hidden marginalized CRFs:

.. code-block:: shell

  dasa_sentiment -v train -t lmcrf -r bhatia data/PotTS/train/\*.json  data/PotTS/dev/\*.json
  dasa_sentiment -v test data/PotTS/test/\*.json > data/PotTS/predicted/lmcrf/lmcrf.json
  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/lmcrf/lmcrf.json


Results
~~~~~~~

.. comment: PotTS
   General Statistics:
   precision    recall  f1-score   support
   positive       0.76      0.79      0.77       437
   negative       0.63      0.49      0.55       209
   neutral       0.69      0.75      0.72       360
   avg / total       0.71      0.71      0.71      1006
   Macro-Averaged F1-Score (Positive and Negative Classes): 66.23%
   Micro-Averaged F1-Score (All Classes): 71.2724%

.. comment: SB10k
   General Statistics
   precision    recall  f1-score   support
   positive       0.64      0.69      0.66       190
   negative       0.44      0.45      0.45       113
   neutral       0.82      0.79      0.80       447
   avg / total       0.72      0.71      0.71       750
   Macro-Averaged F1-Score (Positive and Negative Classes): 55.62%
   Micro-Averaged F1-Score (All Classes): 71.2000%

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| PotTS     |  0.76  | 0.79 |      0.77     |  0.63  | 0.49 |     0.55      |  0.69  | 0.75 |     0.72    |       0.662       |      0.713        |
| SB10k     |  0.64  | 0.69 |      0.66     |  0.44  | 0.45 |     0.45      |  0.82  | 0.79 |     0.8     |       0.556       |      0.712        |
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

References
----------

.. [BHATIA] Parminder Bhatia, Yangfeng Ji, and Jacob
         Eisenstein. 2015. Better Document-Level Sentiment Analysis
         from RST Discourse Parsing. In Proceedings of Empirical
         Methods for Natural Language Processing (EMNLP), Lisbon,
         September.
.. [BOHNET] Bernd Bohnet. 2009. Effiient parsing of syntactic and
	    semantic dependency structures. In Hajic, J., editor,
	    Proceedings of the Thirteenth Conference on Computational
	    Natural Lan- guage Learning: Shared Task, CoNLL 2009,
	    Boulder, Colorado, USA, June 4, 2009 , pages 67--72. ACL.
.. [SIDARENKA] Uladzimir Sidarenka, Tatjana Schefflr and Manfred
	 Stede. 2013.  Rule-based normalization of German Twitter
	 messages. In Language Processing and Knowledge in the Web -
	 25th International Conference, GSCL 2013: Proceedings of the
	 workshop Verarbeitung und Annotation von Sprachdaten aus
	 Genres internetbasierter Kommunikation , Darmstadt, Germany.
.. [WANG] Fei Wang, Yunfang Wu and Likun Qiu. (2013). Exploiting
	  hierarchical discourse structure for review sentiment
	  analysis. In 2013 International Conference on Asian Language
	  Processing, IALP 2013, Urumqi, China, August 17-19, 2013 ,
	  pages 121--124. IEEE.
