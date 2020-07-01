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

Description
-----------

This package provides implementations of various discourse-aware
sentiment analysis (DASA) methods.  Most of these approaches infer the
overall polarity of the input (*e.g,* of a review) from the polarity
scores of its elementary discourse units (EDUs) by either accumulating
these scores over the RST tree or just considering a single EDU, which
is considered to be most representative of the whole analyzed text.

Data Preparation
----------------

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

    ./scripts/enrich_json data/SST/sst.json data/IMDB/{pos,neg}/*.json


* **discourse segmentation** was done with a slightly adjusted
  `version <https://github.com/WladimirSidorenko/NeuralEDUSeg/tree/master/scripts>`_
  of `Wang et al.`_ 's neural discourse segmenter::

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

    using the code from repository https://github.com/WladimirSidorenko/SO-CAL

  * **XLNet** scores were computed with the script
    `add_polarity_scores <scripts/add_polarity_scores>`_ using included
    in this repo::

      ./scripts/add_polarity_scores data/IMDB/{pos,neg}/*.json
      ./scripts/add_polarity_scores data/SST/raw/sst.json

Models
------

DDR
^^^

To determine the polarity of a tweet using the discourse depth
reweighting (DDR) method [BHATIA]_, you can use the following command
to create the model:

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

.. comment: IMDB (So-Cal)

.. comment: SST (So-Cal)

.. comment: IMDB (XLNET)

.. comment: SST (XLNET)

+-----------+--------------------+---------------------+--------------------+------------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy     |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+

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

.. comment: IMDB (So-Cal)
   Command: dasa_sentiment cv -t last -n 3 -s socal  data/IMDB/*/*.json
   precision_macro: 0.3518 (+/- 0.04)
   recall_macro: 0.1267 (+/- 0.04)
   f1_macro: 0.1840 (+/- 0.05)
   accuracy: 0.3185 (+/- 0.06)

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

   Command: dasa_sentiment cv -t last -n 3 -s xlnet  data/IMDB/*/*.json
   precision_macro: 0.4500 (+/- 0.10)
   recall_macro: 0.3003 (+/- 0.24)
   f1_macro: 0.2679 (+/- 0.22)
   accuracy: 0.5005 (+/- 0.44)

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

+-----------+--------------------+---------------------+--------------------+----------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy   |
+-----------+--------------------+---------------------+--------------------+----------------+
|                                               So-Cal                                       |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |  0.3518 (+/- 0.04) |  0.1267 (+/- 0.04)  |  0.1840 (+/- 0.05) | 31.85 (+/- 6)  |
| SST       |  0.4484            |  0.4253             |  0.4168            | 43.2%          |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                        |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |  0.45 (+/- 0.1)    |  0.3003 (+/- 0.24)  |  0.2679 (+/- 0.22) | 50.05 (+/- 44) |
| SST       |  0.4007            |  0.36               |  0.3141            | 33.54%         |
+-----------+--------------------+---------------------+--------------------+----------------+


No-Discourse
^^^^^^^^^^^^

To predict the polarity of a tweet irrespective of discourse
information, you can invoke the above scripts as follows:

.. code-block:: shell

  dasa_sentiment train -t no-discourse -m data/SST/models/no-discourse.model -n 3 \
  -s xlnet -d data/SST/dev/dev.json data/SST/train/train.json

and then the following scripts to predict the label and evaluate the
quality:

.. code-block:: shell

  dasa_sentiment test -m data/SST/models/no-discourse.model data/SST/test/test.json \
  > data/SST/predicted/no-discourse/no-discourse.xlnet.json
  dasa_evaluate data/SST/test/test.json data/SST/predicted/no-discourse/no-discourse.json

equivalently for IMDB:

.. code-block:: shell

  dasa_sentiment -v train -t no-discourse data/SB10k/train/\*.json  data/SB10k/dev/\*.json
  dasa_sentiment -v test data/SB10k/test/\*.json > data/SB10k/predicted/no-discourse/no-discourse.json
  dasa_evaluate data/SB10k/test/ data/SB10k/predicted/no-discourse/no-discourse .json


Results
~~~~~~~

.. comment: IMDB (So-Cal)
   dasa_sentiment cv -t no-discourse -n 3 -s socal  data/IMDB/*/*.json 
   precision_macro: 0.5496 (+/- 0.10)
   recall_macro: 0.4475 (+/- 0.16)
   f1_macro: 0.4852 (+/- 0.13)
   accuracy: 0.7490 (+/- 0.14)

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
   Command: dasa_sentiment cv -t no-discourse -n 3 -s xlnet  data/IMDB/*/*.json
   precision_macro: 0.5620 (+/- 0.12)
   recall_macro: 0.4832 (+/- 0.16)
   f1_macro: 0.5168 (+/- 0.14)
   accuracy: 0.8060 (+/- 0.07)

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

+-----------+--------------------+---------------------+--------------------+------------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy     |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |   0.5496 (+/- 0.1) |  0.4475 (+/- 0.16)  |  0.4852 (+/- 0.13) |  74.9 (+/- 0.14) |
| SST       |   0.4829           |  0.4917             |  0.4797            |  58.247          |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |   0.562 (+/- 0.12) |  0.4832 (+/- 0.16)  |  0.5168 (+/- 0.14) |  80.6 (+/- 0.07) |
| SST       |   0.6848           |  0.6854             |  0.6842            |  75.569          |
+-----------+--------------------+---------------------+--------------------+------------------+

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

.. comment: IMDB (So-Cal)

.. comment: SST (So-Cal)
   General Statistics:
              precision    recall  f1-score   support

    negative       0.55      0.38      0.45       606
    neutral        0.18      0.39      0.25       254
    positive       0.61      0.51      0.55       589

   accuracy                            0.43      1449
   macro avg       0.45      0.43      0.42      1449
   weighted avg    0.51      0.43      0.46      1449


   Macro-Averaged Precision: 44.84%
   Macro-Averaged Recall: 42.53%
   Macro-Averaged F1-Score: 41.68%
   Micro-Averaged F1-Score (All Classes): 43.2022%

.. comment: IMDB (XLNET)

.. comment: SST (XLNET)
   General Statistics:
              precision    recall  f1-score   support

    negative       0.46      0.48      0.47       606
    neutral        0.17      0.47      0.25       254
    positive       0.56      0.13      0.21       589

    accuracy                           0.34      1449
    macro avg      0.40      0.36      0.31      1449
    weighted avg   0.45      0.34      0.33      1449

    Macro-Averaged Precision: 40.07%
    Macro-Averaged Recall: 36.00%
    Macro-Averaged F1-Score: 31.14%
    Micro-Averaged F1-Score (All Classes): 33.5404%


+-----------+--------------------+---------------------+--------------------+------------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy     |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+

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

.. comment: IMDB (So-Cal)

.. comment: SST (So-Cal)

.. comment: IMDB (XLNET)

.. comment: SST (XLNET)

+-----------+--------------------+---------------------+--------------------+------------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy     |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+

RDM
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

.. comment: IMDB (So-Cal)

.. comment: SST (So-Cal)

.. comment: IMDB (XLNET)

.. comment: SST (XLNET)

+-----------+--------------------+---------------------+--------------------+------------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy     |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+

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

.. comment: IMDB (So-Cal)

.. comment: SST (So-Cal)

.. comment: IMDB (XLNET)

.. comment: SST (XLNET)

+-----------+--------------------+---------------------+--------------------+------------------+
| **Data**  |  Macro-Precision   |     Macro-Recall    |  :math:`Macro F_1` |     Accuracy     |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               So-Cal                                         |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+
|                                               XLNET                                          |
+-----------+--------------------+---------------------+--------------------+------------------+
| IMDB      |                    |                     |                    |                  |
| SST       |                    |                     |                    |                  |
+-----------+--------------------+---------------------+--------------------+------------------+

.. _IMDB: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
.. _Stanford Sentiment Treebank: http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
.. _imdb2json: scripts/imdb2json
.. _sst2json: scripts/sst2json
.. _enrich_json: scripts/enrich_json
.. _`Wang et al.`: https://arxiv.org/abs/1808.09147

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
