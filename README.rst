Discourse-Aware Sentiment Analysis
==================================

.. image:: https://travis-ci.org/WladimirSidorenko/DASA.svg?branch=master
   :alt: Build Status
   :align: right
   :target: https://travis-ci.org/WladimirSidorenko/DASA

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :alt: MIT License
   :align: right
   :target: http://opensource.org/licenses/MIT


.. contents::
  :local:

Description
-----------

This package provides implementations of various discourse-aware
sentiment analysis (DASA) methods.  Most of these approaches infer the
overall polarity of the input (*e.g,* of a review) from the polarity
scores of its elementary discourse units (EDUs) by either accumulating
these scores over the RST tree or just considering a single EDU, which
is considered to be most representative of the whole analyzed text.

Data
----

We use the `IMDB`_ corpus and `Stanford Sentiment Treebank`_ as our
primary data for evaluation.

The exact preparation steps for these datasets looked as follows:

* First, the original files of these corpora were **converted to the
  JSON format** using the scripts `imdb2json`_ and `sst2json`_, which
  are also included in this repository::

    ./scripts/sst2json data/SST/ > data/SST/sst.json
    ./scripts/imdb2json data/IMDB/*/*.txt


* Afterwards, converted data were **enriched** with information about
  lemma, PoS tag, dependency relation, and morphological features
  using the provided script `enrich_json`_::

    ./scripts/enrich_json data/SST/sst.json data/IMDB/*/*.json


* **discourse segmentation** was done with a slightly adjusted
  `version
  <https://github.com/WladimirSidorenko/NeuralEDUSeg/tree/master/scripts>`_
  of `Wang et al. <https://arxiv.org/abs/1808.09147>`_ 's neural
  discourse segmenter::

    git clone git@github.com:WladimirSidorenko/NeuralEDUSeg.git
    cd NeuralEDUSeg
    python -m venv venv
    . venv/bin/activate
    pip install -e .
    dsegment ../DASA/data/IMDB/*/*.json ../DASA/data/SST/sst.json

* In a similar way, **discourse parsing** was performed with a
  slightly adjusted `version
  <https://github.com/WladimirSidorenko/StageDP>`_ of two-stage
  discourse parser by `Wang et
  al. <https://www.aclweb.org/anthology/P17-2029/>`_::

    git clone git@github.com:WladimirSidorenko/StageDP.git
    cd StageDP
    python -m venv venv
    . venv/bin/activate
    pip install -e .
    dparse ../DASA/data/IMDB/*/*.json ../DASA/data/SST/sst.json

* **sentiment scores**

  * **SO-Cal** scores were added using the following commands::

      ./Source_Code/sentiment_calculator/SO_JsonCalc.py ~/Projects/DASA/data/SST/sst.json
      ./Source_Code/sentiment_calculator/SO_JsonCalc.py ~/Projects/DASA/data/IMDB/*/*.json

    using the code from the `SO-CAL repository
    <https://github.com/WladimirSidorenko/SO-CAL>`_

  * **XLNet** scores were computed with the script
    `add_polarity_scores <scripts/add_polarity_scores>`_ using included
    in this repo::

      ./scripts/add_polarity_scores data/IMDB/{pos,neg}/*.json
      ./scripts/add_polarity_scores data/SST/raw/sst.json

Models
------

This package comes with the following discourse-aware sentiment models:

* **no-discourse**, which simply re-uses the classification of the
  base predictor

* **last**, which decides on the polarity of the whole document by the
  polarity of the last elementary discourse unit

* **root**, which induce the polarity of the whole document from the
  polarity of the top-most (root) EDU

* **ddr** |--| a re-implementation of the discourse depth reweighting
  (DDR) approach by [BHATIA]_

* **r2n2** |--| a re-implementation of the rhetorical recursive
  neural-network approach by [BHATIA]_

* **rdm** |--| our own **Recursive Dirichlet Model**.

Training
--------

To train a discourse-aware sentiment analysis model on the `Stanford
Sentiment Treebank`_, you can use the following command::

  dasa_sentiment train -t $MODEL -m data/SST/models/${MODEL}.${SSCORE}.model \
  -n 3 -s ${SSCORE} -d data/SST/dev/dev.json data/SST/train/train.json

where `${MODEL}` is one of the implemented models (`rdm`,
`no-discourse`, `root`, `last`, `wang`, `ddr`, or `r2n2`); and
`${SSCORE}` is one of the available sentiment scores in the respective
data set (`xlnet` or `socal`).

Testing and Evaluation
----------------------

Once you've trained your model, you can run it on the `SST test set
<data/SST/test/test.json>`_::

  dasa_sentiment -v test -m data/SST/models/${MODEL}.${SSCORE}.model \
  data/SST/test/test.json > data/SST/predicted/${MODEL}/${MODEL}.${SSCORE}.json

and evaluate the predictions with the help of the provided
`dasa_evaluate <scripts/dasa_evaluate>`_ script::

  dasa_evaluate data/PotTS/test/ data/PotTS/predicted/${MODEL}/${MODEL}.${SSCORE}.json

For data sets that do not have an explicit test set, you can
cross-validate your classifier with the following command::

  dasa_sentiment cv -t ${MODEL} -n 2 -s ${SSCORE} PATH_TO_THE_DATA_FILES

for example::

  dasa_sentiment cv -t rdm -n 2 -s xlnet data/IMDB/*/*.json


Results
-------

DDR
^^^

.. comment: IMDB (So-Cal)
   Command:
     dasa_sentiment -v cv -t ddr -n2 -s socal data/IMDB/{pos,neg}/*.json
   Results:
     precision_macro: 0.5465 (+/- 0.09)
     recall_macro: 0.3415 (+/- 0.19)
     f1_macro: 0.4076 (+/- 0.16)
     accuracy: 53.7500 (+/- 10.57)


.. comment: SST (So-Cal)
   Command:
     dasa_sentiment train -t ddr -n3 -m data/models/ddr.socal.model -s socal -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment test -m data/models/ddr.socal.model data/SST/test/test.json > data/SST/predicted/ddr/ddr.socal.json
     dasa_evaluate data/SST/test/test.json data/SST/predicted/ddr/ddr.socal.json
   Results:
              precision    recall  f1-score   support

    negative       0.62      0.50      0.55       606
     neutral       0.20      0.35      0.25       254
    positive       0.67      0.60      0.63       589

    accuracy                           0.51      1449
    macro avg      0.50      0.48      0.48      1449
    weighted avg   0.57      0.51      0.53      1449

    Macro-Averaged Precision: 49.92%
    Macro-Averaged Recall: 48.19%
    Macro-Averaged F1-Score: 48.11%
    Micro-Averaged F1-Score (All Classes): 51.3458%

.. comment: IMDB (XLNET)
   Command:
     dasa_sentiment -v cv -t ddr -n2 -s xlnet data/IMDB/{pos,neg}/*.json
   Results:
     precision_macro: 0.4956 (+/- 0.01)
     recall_macro: 0.2978 (+/- 0.17)
     f1_macro: 0.3291 (+/- 0.15)
     accuracy: 49.8500 (+/- 29.08)

.. comment: SST (XLNET)
   Command:
     dasa_sentiment train -t ddr -n3 -m data/models/ddr.xlnet.model -s xlnet -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment test -m data/models/ddr.xlnet.model data/SST/test/test.json > data/SST/predicted/ddr/ddr.xlnet.json
     dasa_evaluate data/SST/test/test.json data/SST/predicted/ddr/ddr.xlnet.json
   Results:
              precision    recall  f1-score   support

    negative       0.48      0.48      0.48       606
     neutral       0.00      0.00      0.00       254
    positive       0.46      0.65      0.54       589

    accuracy                           0.47      1449
   macro avg       0.31      0.38      0.34      1449
   weighted avg    0.39      0.47      0.42      1449

   Macro-Averaged Precision: 31.34%
   Macro-Averaged Recall: 37.74%
   Macro-Averaged F1-Score: 33.96%
   Micro-Averaged F1-Score (All Classes): 46.5839%

+-----------+--------------------+---------------------+--------------------+-------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |      Macro-F1      |     Accuracy      |
+===========+====================+=====================+====================+===================+
|                                               So-Cal                                          |
+-----------+--------------------+---------------------+--------------------+-------------------+
| IMDB      |  0.5465 (+/- 0.09) |  0.3415 (+/- 0.19)  |  0.4076 (+/- 0.16) | 53.75 (+/- 10.57) |
+-----------+--------------------+---------------------+--------------------+-------------------+
| SST       |      0.4992        |       0.4819        |       0.4811       |    51.3458        |
+-----------+--------------------+---------------------+--------------------+-------------------+
|                                               XLNET                                           |
+-----------+--------------------+---------------------+--------------------+-------------------+
| IMDB      |  0.4956 (+/- 0.01) |  0.2978 (+/- 0.17)  |  0.3291 (+/- 0.15) | 49.85 (+/- 29.08) |
+-----------+--------------------+---------------------+--------------------+-------------------+
| SST       |      0.3134        |       0.3774        |       0.3396       |    46.5839        |
+-----------+--------------------+---------------------+--------------------+-------------------+

Last EDU
^^^^^^^^

.. comment: IMDB (So-Cal)
   Command:
     dasa_sentiment cv -t last -n 2 -s socal  data/IMDB/*/*.json
   Results
     precision_macro: 0.5229 (+/- 0.05)
     recall_macro: 0.3535 (+/- 0.15)
     f1_macro: 0.4003 (+/- 0.11)
     accuracy: 58.9500 (+/- 18.67)

.. comment: SST (So-Cal)
   General Statistics:
               precision    recall  f1-score   support

   negative       0.55      0.38      0.45       606
   neutral        0.18      0.39      0.25       254
   positive       0.61      0.51      0.55       589

   accuracy                           0.43      1449
   macro avg      0.45      0.43      0.42      1449
   weighted avg   0.51      0.43      0.46      1449

   Macro-Averaged Precision: 44.84%
   Macro-Averaged Precision: 42.53%
   Macro-Averaged F1-Score: 41.68%
   Micro-Averaged F1-Score (All Classes): 43.2022%

.. comment: IMDB (XLNET)

   Command: dasa_sentiment cv -t last -n 2 -s xlnet  data/IMDB/*/*.json
   precision_macro: 0.4500 (+/- 0.10)
   recall_macro: 0.3003 (+/- 0.24)
   f1_macro: 0.2679 (+/- 0.22)
   accuracy: 50.05 (+/- 44.44)

.. comment: SST (XLNET)
   Train Command: dasa_sentiment train -t last -m data/SST/models/last.xlnet.model -n 3 -s xlnet -d data/SST/dev/dev.json data/SST/train/train.json
   Test Command: dasa_sentiment test -m data/SST/models/last.xlnet.model data/SST/test/test.json > data/SST/predicted/last/last.xlnet.json
   General Statistics:
                 precision    recall  f1-score   support

    negative       0.46      0.48      0.47       606
    neutral        0.17      0.47      0.25       254
    positive       0.56      0.13      0.21       589

    accuracy                           0.34      1449
    macro av       0.40      0.36      0.31      1449
    weighted avg   0.45      0.34      0.33      1449

    Macro-Averaged Precision: 40.07%
    Macro-Averaged Recall: 36.00%
    Macro-Averaged F1-Score: 31.14%
    Micro-Averaged F1-Score (All Classes): 33.5404%

+-----------+--------------------+---------------------+---------------------+-------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |      Macro-F1       |      Accuracy     |
+===========+====================+=====================+=====================+===================+
|                                                So-Cal                                          |
+-----------+--------------------+---------------------+---------------------+-------------------+
| IMDB      |  0.5229 (+/- 0.05) |  0.3535 (+/- 0.15)  |  0.4003 (+/- 0.11)  | 58.95 (+/- 18.67) |
+-----------+--------------------+---------------------+---------------------+-------------------+
| SST       |  0.4484            |  0.4253             |  0.4168             | 43.2022           |
+-----------+--------------------+---------------------+---------------------+-------------------+
|                                               XLNET                                            |
+-----------+--------------------+---------------------+---------------------+-------------------+
| IMDB      |  0.45 (+/- 0.1)    |  0.3003 (+/- 0.24)  |  0.2679 (+/- 0.22)  | 50.05 (+/- 44.44) |
+-----------+--------------------+---------------------+---------------------+-------------------+
| SST       |  0.4007            |  0.36               |  0.3141             | 33.5404           |
+-----------+--------------------+---------------------+---------------------+-------------------+


No-Discourse
^^^^^^^^^^^^

.. comment: IMDB (So-Cal)
   Command
     dasa_sentiment cv -t no-discourse -n 2 -s socal  data/IMDB/*/*.json
   Results
     precision_macro: 0.5496 (+/- 0.10)
     recall_macro: 0.4475 (+/- 0.16)
     f1_macro: 0.4852 (+/- 0.13)
     accuracy: 74.9000 (+/- 13.99)

.. comment: SST (So-Cal)
   General Statistics:
               precision   recall  f1-score   support

   negative       0.60      0.63      0.61       606
   neutral        0.22      0.11      0.14       254
   positive       0.64      0.74      0.68       589

   accuracy                           0.58      1449
   macro avg      0.48      0.49      0.48      1449
   weighted avg   0.55      0.58      0.56      1449

   Macro-Averaged Precision: 48.29%
   Macro-Averaged Precision: 49.17%
   Macro-Averaged F1-Score: 47.97%
   Micro-Averaged F1-Score (All Classes): 58.2471%

.. comment: IMDB (XLNET)
   Command
     dasa_sentiment cv -t no-discourse -n 2 -s xlnet  data/IMDB/*/*.json

   Results
     precision_macro: 0.5620 (+/- 0.12)
     recall_macro: 0.4832 (+/- 0.16)
     f1_macro: 0.5168 (+/- 0.14)
     accuracy: 80.6000 (+/- 6.60)

.. comment: SST (XLNET)
   General Statistics:
               precision    recall  f1-score   support

   negative       0.84      0.77      0.81       606
   neutral        0.38      0.39      0.38       254
   positive       0.84      0.90      0.87       589

   accuracy                           0.76      1449
   macro avg      0.68      0.69      0.68      1449
   weighted avg   0.76      0.76      0.76      1449

   Macro-Averaged Precision: 68.48%
   Macro-Averaged Precision: 68.54%
   Macro-Averaged F1-Score: 68.42%
   Micro-Averaged F1-Score (All Classes): 75.5694%

+-----------+--------------------+---------------------+---------------------+------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |       Macro-F1      |     Accuracy     |
+===========+====================+=====================+=====================+==================+
|                                               So-Cal                                          |
+-----------+--------------------+---------------------+---------------------+------------------+
| IMDB      |   0.5496 (+/- 0.1) |  0.4475 (+/- 0.16)  |  0.4852 (+/- 0.13)  | 74.9 (+/- 13.99) |
+-----------+--------------------+---------------------+---------------------+------------------+
| SST       |   0.4829           |  0.4917             |  0.4797             |  58.247          |
+-----------+--------------------+---------------------+---------------------+------------------+
|                                               XLNET                                           |
+-----------+--------------------+---------------------+---------------------+------------------+
| IMDB      |   0.562 (+/- 0.12) |  0.4832 (+/- 0.16)  |  0.5168 (+/- 0.14)  |  80.6 (+/- 6.6)  |
+-----------+--------------------+---------------------+---------------------+------------------+
| SST       |   0.6848           |  0.6854             |  0.6842             |  75.569          |
+-----------+--------------------+---------------------+---------------------+------------------+

Root EDU
^^^^^^^^

.. comment: IMDB (So-Cal)
   Command:
     dasa_sentiment cv -t root -n 2 -s socal data/IMDB/*/*.json

   Results:
     precision_macro: 0.5173 (+/- 0.03)
     recall_macro: 0.3450 (+/- 0.13)
     f1_macro: 0.4036 (+/- 0.10)
     accuracy: 57.3500 (+/- 10.46)

.. comment: SST (So-Cal)
   Commands:
     dasa_sentiment train -t root -m data/SST/models/root.socal.model -n 3 -s socal -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment -v test -m data/SST/models/root.socal.model  data/SST/test/test.json > data/SST/predicted/root/root.socal.json
     dasa_evaluate  data/SST/test/test.json data/SST/predicted/root/root.socal.json

   General Statistics:
                precision    recall  f1-score   support
   negative       0.62      0.41      0.49       606
   neutral        0.18      0.44      0.26       254
   positive       0.66      0.49      0.56       589

   accuracy                           0.45      1449
   macro avg       0.49      0.45      0.44      1449
   weighted avg       0.56      0.45      0.48      1449

   Macro-Averaged Precision: 48.60%
   Macro-Averaged Recall: 44.70%
   Macro-Averaged F1-Score: 43.79%
   Micro-Averaged F1-Score (All Classes): 44.7895%

.. comment: IMDB (XLNET)
   Command:
     dasa_sentiment cv -t root -n 2 -s xlnet data/IMDB/*/*.json

   Results:
     precision_macro: 0.4501 (+/- 0.40)
     recall_macro: 0.4002 (+/- 0.37)
     f1_macro: 0.3676 (+/- 0.37)
     accuracy: 50.0000 (+/- 44.67)

.. comment: SST (XLNET)
   Commands:
     dasa_sentiment train -t root -m data/SST/models/root.xlnet.model -n 3 -s socal -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment -v test -m data/SST/models/root.socal.model  data/SST/test/test.json > data/SST/predicted/root/root.socal.json
     dasa_evaluate  data/SST/test/test.json data/SST/predicted/root/root.socal.json

   General Statistics:
                precision  recall  f1-score   support

   negative       0.48      0.41      0.45       606
   neutral        0.17      0.54      0.26       254
   positive       0.62      0.13      0.21       589

   accuracy                           0.32      1449
   macro avg      0.43      0.36      0.30      1449
   weighted avg   0.48      0.32      0.32      1449

   Macro-Averaged Precision: 42.52%
   Macro-Averaged Recall: 36.10%
   Macro-Averaged F1-Score: 30.48%
   Micro-Averaged F1-Score (All Classes): 31.9531%

+-----------+--------------------+---------------------+--------------------+-------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |       Macro-F1     |      Accuracy     |
+===========+====================+=====================+====================+===================+
|                                               So-Cal                                          |
+-----------+--------------------+---------------------+--------------------+-------------------+
| IMDB      | 0.5173 (+/- 0.03)  |  0.3450 (+/- 0.13)  |  0.4036 (+/- 0.1)  | 57.35 (+/- 10.46) |
+-----------+--------------------+---------------------+--------------------+-------------------+
| SST       | 0.486              |  0.447              |  0.4379            | 44.7895           |
+-----------+--------------------+---------------------+--------------------+-------------------+
|                                               XLNET                                           |
+-----------+--------------------+---------------------+--------------------+-------------------+
| IMDB      | 0.4501 (+/- 0.40)  |  0.4002 (+/- 0.37)  |  0.3676 (+/- 0.37) | 50.0 (+/- 44.67)  |
+-----------+--------------------+---------------------+--------------------+-------------------+
| SST       | 0.4252             |  0.361              |  0.3048            | 31.9531           |
+-----------+--------------------+---------------------+--------------------+-------------------+

R2N2
^^^^

.. comment: IMDB (So-Cal)
   Command:
     dasa_sentiment cv -t r2n2 -s socal -n 2 data/IMDB/{pos,neg}/*.json
   Results:
     precision_macro: 0.4624 (+/- 0.19)
     recall_macro: 0.3649 (+/- 0.22)
     f1_macro: 0.4022 (+/- 0.20)
     accuracy: 71.3500 (+/- 8.14)

.. comment: SST (So-Cal)
   General Statistics:
   precision    recall  f1-score   support

   negative       0.62      0.61      0.62       606
   neutral       0.23      0.11      0.14       254
   positive       0.61      0.76      0.67       589

   accuracy                           0.58      1449
   macro avg       0.49      0.49      0.48      1449
   weighted avg       0.55      0.58      0.56      1449

   Macro-Averaged Precision: 48.51%
   Macro-Averaged Recall: 49.13%
   Macro-Averaged F1-Score: 47.83%
   Micro-Averaged F1-Score (All Classes): 58.1781%

.. comment: IMDB (XLNET)
   Command:
     dasa_sentiment cv -t r2n2 -s xlnet -n 2 data/IMDB/{pos,neg}/*.json
   Results:
     precision_macro: 0.4957 (+/- 0.17)
     recall_macro: 0.4061 (+/- 0.21)
     f1_macro: 0.4437 (+/- 0.19)
     accuracy: 75.0000 (+/- 3.31)

.. comment: SST (XLNET)
   General Statistics:
              precision    recall  f1-score   support

    negative       0.83      0.76      0.79       606
     neutral       0.36      0.37      0.36       254
    positive       0.83      0.89      0.86       589

    accuracy                           0.75      1449
    macro avg      0.67      0.67      0.67      1449
    weighted avg   0.75      0.75      0.75      1449

    Macro-Averaged Precision: 67.33%
    Macro-Averaged Recall: 67.39%
    Macro-Averaged F1-Score: 67.26%
    Micro-Averaged F1-Score (All Classes): 74.6032%

+-----------+--------------------+---------------------+--------------------+------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |      Macro-F1      |     Accuracy     |
+===========+====================+=====================+====================+==================+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      | 0.4624 (+/- 0.19)  |  0.3649 (+/- 0.22)  |  0.4022 (+/- 0.2)  | 71.35 (+/- 8.14) |
+-----------+--------------------+---------------------+--------------------+------------------+
| SST       |    0.4851          |    0.4913           |    0.4783          |   58.1781        |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      | 0.4957 (+/- 0.17)  |  0.4061 (+/- 0.21)  |  0.4437 (+/- 0.19) | 75.0 (+/- 3.31)  |
+-----------+--------------------+---------------------+--------------------+------------------+
| SST       |    0.6733          |    0.6739           |    0.6726          |   74.6032        |
+-----------+--------------------+---------------------+--------------------+------------------+

RDM
^^^

.. comment: IMDB (So-Cal)
   Command:
     dasa_sentiment -v cv -t rdp -n2 -s socal data/IMDB/{pos,neg}/*.json
   Results:

.. comment: SST (So-Cal)
   Command:
     dasa_sentiment train -t rdp -n3 -m data/models/rdp.socal.model -s socal -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment test -m data/models/rdp.socal.model data/SST/test/test.json > data/SST/predicted/rdp/rdp.socal.json
     dasa_evaluate data/SST/test/test.json data/SST/predicted/rdp/rdp.socal.json
   Results:

.. comment: IMDB (XLNET)
   Command:
     dasa_sentiment -v cv -t rdp -n2 -s xlnet data/IMDB/{pos,neg}/*.json
   Results:

.. comment: SST (XLNET)
   Command:
     dasa_sentiment train -t rdp -n3 -m data/models/rdp.xlnet.model -s xlnet -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment test -m data/models/rdp.xlnet.model data/SST/test/test.json > data/SST/predicted/rdp/rdp.xlnet.json
     dasa_evaluate data/SST/test/test.json data/SST/predicted/rdp/rdp.xlnet.json
   Results:

+-----------+--------------------+---------------------+--------------------+------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |      Macro-F1      |     Accuracy     |
+===========+====================+=====================+====================+==================+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+

WANG
^^^^

.. comment: IMDB (So-Cal)
   Command:
     dasa_sentiment -v cv -t wang -n2 -s socal data/IMDB/{pos,neg}/*.json
   Results:
     precision_macro: 0.3297 (+/- 0.01)
     recall_macro: 0.1305 (+/- 0.10)
     f1_macro: 0.1678 (+/- 0.09)
     accuracy: 29.9500 (+/- 13.51)

.. comment: SST (So-Cal)
   Command:
     dasa_sentiment train -t wang -n3 -m data/models/wang.socal.model -s socal -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment test -m data/models/wang.socal.model data/SST/test/test.json > data/SST/predicted/wang/wang.socal.json
     dasa_evaluate data/SST/test/test.json data/SST/predicted/wang/wang.socal.json
   Results:
                 precision    recall  f1-score   support
     negative       0.55      0.37      0.44       606
     neutral        0.19      0.38      0.25       254
     positive       0.56      0.51      0.53       589

     accuracy                           0.43      1449
     macro avg      0.43      0.42      0.41      1449
     weighted avg   0.49      0.43      0.45      1449

     Macro-Averaged Precision: 43.45%
     Macro-Averaged Recall:    41.81%
     Macro-Averaged F1-Score:  40.83%
     Micro-Averaged F1-Score (All Classes): 42.5811%

.. comment: IMDB (XLNET)
   Command:
     dasa_sentiment -v cv -t wang -n2 -s xlnet data/IMDB/{pos,neg}/*.json
   Results:
     precision_macro: 0.3833 (+/- 0.10)
     recall_macro: 0.2992 (+/- 0.24)
     f1_macro: 0.2670 (+/- 0.22)
     accuracy: 49.9000 (+/- 44.27)

.. comment: SST (XLNET)
   Command:
     dasa_sentiment train -t wang -n3 -m data/models/wang.xlnet.model -s xlnet -d data/SST/dev/dev.json data/SST/train/train.json
     dasa_sentiment test -m data/models/wang.xlnet.model data/SST/test/test.json > data/SST/predicted/wang/wang.xlnet.json
     dasa_evaluate data/SST/test/test.json data/SST/predicted/wang/wang.xlnet.json
   Results:
                 precision   recall  f1-score   support
     negative       0.44      0.34      0.39       606
     neutral        0.18      0.48      0.26       254
     positive       0.42      0.20      0.27       589

     accuracy                           0.31      1449
     macro avg      0.34      0.34      0.31      1449
     weighted avg   0.38      0.31      0.32      1449

     Macro-Averaged Precision: 34.48%
     Macro-Averaged Recall:    34.24%
     Macro-Averaged F1-Score:  30.59%
     Micro-Averaged F1-Score (All Classes): 31.0559%

+-----------+--------------------+---------------------+--------------------+------------------+
|   Data    |  Macro-Precision   |     Macro-Recall    |      Macro-F1      |     Accuracy     |
+===========+====================+=====================+====================+==================+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |  0.3297 (+/- 0.01) |  0.1305 (+/- 0.1)   |  0.1678 (+/- 0.09) | 29.95 (+/- 13.51)|
+-----------+--------------------+---------------------+--------------------+------------------+
| SST       |      0.4345        |       0.4181        |       0.4083       |     42.5811      |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |  0.3833 (+/- 0.1)  |  0.2992 (+/- 0.24)  |  0.267 (+/- 0.22)  | 49.9 (+/- 44.27) |
+-----------+--------------------+---------------------+--------------------+------------------+
| SST       |      0.3448        |       0.3424        |       0.3059       |     31.0559      |
+-----------+--------------------+---------------------+--------------------+------------------+

.. |--| unicode:: U+2013   .. en dash
.. |---| unicode:: U+2014  .. em dash, trimming surrounding whitespace
   :trim:

.. _IMDB: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
.. _Stanford Sentiment Treebank: http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
.. _imdb2json: scripts/imdb2json
.. _sst2json: scripts/sst2json
.. _enrich_json: scripts/enrich_json

References
----------

.. [BHATIA] Parminder Bhatia, Yangfeng Ji, and Jacob
         Eisenstein. 2015. Better Document-Level Sentiment Analysis
         from RST Discourse Parsing. In Proceedings of Empirical
         Methods for Natural Language Processing (EMNLP), Lisbon,
         September.
.. [WANG] Fei Wang, Yunfang Wu and Likun Qiu. (2013). Exploiting
	  hierarchical discourse structure for review sentiment
	  analysis. In 2013 International Conference on Asian Language
	  Processing, IALP 2013, Urumqi, China, August 17-19, 2013 ,
	  pages 121--124. IEEE.


param = dict(
    root='/home/sidorenko/Work/lab-results-conversion/nutrient-bag-ids+blacklisted',
    data='/home/sidorenko/Work/lab-results-conversion/nutrient-bag-ids+blacklisted',
    timestamp='2020-07-31',
    interpolate='smooth',
    whiten=False,
    clean=None,
    impute=None,
    depth='any',
    protos='/home/sidorenko/Work/lab-results-conversion/nutrient-bag-ids+blacklisted/protobufs')
