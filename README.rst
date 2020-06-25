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

This package provides several implementations of common
discourse-aware sentiment analysis (DASA) methods.  Most of these
approaches infer the overall polarity of the input (*e.g,* of a tweet)
from the polarity scores of its elementary discourse units (EDUs) by
either accumulating these scores over the RST tree or choosing a
single EDU, which is most representative of the whole analyzed text
(*e.g.*, the last discourse segment).

Data Preparation
----------------

We use the `IMDB`_ corpus and `Stanford Sentiment Treebank`_ as our
primary data for evaluation.

The exact preparation steps for these datasets looked as follows:

* First, I have converted the original files of these corpora to the
  **JSON format** using the scripts `scripts/imdbjson` and
  `scripts/sst2json`, which are also included in this repository::

    ./scripts/sst2json data/SST/ > data/SST/sst.json
    ./scripts/imdb2json data/IMDB/*/*.txt


* Afterwards, I have enriched these converted data with information
  about lemma, PoS tag, dependency relation, and morphological
  features using the provided script `./scripts/enrich_json`::

    ./scripts/enrich_json data/SST/sst.json data/IMDB/{pos,neg}/*.json


* **Discourse Segmentation**

  Discourse segmentation was done with a [slightly adjusted
  version](https://github.com/WladimirSidorenko/NeuralEDUSeg/tree/master/scripts)
  of Wang et al.'s [neural discourse
  segmenter](https://arxiv.org/abs/1808.09147)::
    git clone git@github.com:WladimirSidorenko/NeuralEDUSeg.git
    cd NeuralEDUSeg
    python -m venv venv
    . venv/bin/activate
    pip install -e .
    dsegment ../DASA/data/IMDB/*/*.json ../DASA/data/SST/sst.json

* **Discourse Parsing**

  Similarly, discourse parsing was performed with a [slightly adjusted
  version](https://github.com/WladimirSidorenko/StageDP) of two-stage
  discourse parser by [Yizhong Wang et
  al.](https://www.aclweb.org/anthology/P17-2029/)::
    git clone git@github.com:WladimirSidorenko/StageDP.git
    cd StageDP
    python -m venv venv
    . venv/bin/activate
    pip install -e .
    dparse ../DASA/data/IMDB/*/*.json ../DASA/data/SST/sst.json

* **Sentiment Scores**

  * SO-Cal sentiment scores where added using the following commands::
      ./Source_Code/sentiment_calculator/SO_JsonCalc.py ~/Projects/DASA/data/SST/sst.json
      ./Source_Code/sentiment_calculator/SO_JsonCalc.py ~/Projects/DASA/data/IMDB/*/*.json
    at commit `419bc23` of (this project)[https://github.com/WladimirSidorenko/SO-CAL]

Examples
--------

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

.. comment: IMDB

.. comment: SST

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |        |      |               |        |      |               |        |      |             |                   |                   |
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

.. comment: IMDB

.. comment: SST

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |        |      |               |        |      |               |        |      |             |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+


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

.. comment: IMDB

.. comment: SST (XLNET)
   General Statistics:
                precision   recall  f1-score   support

   positive       0.84      0.90      0.87       589
   negative       0.84      0.77      0.81       606
   neutral        0.38      0.39      0.38       254

   accuracy                            0.76      1449
   macro avg       0.68      0.69      0.68      1449
   weighted avg    0.76      0.76      0.76      1449

   Macro-Averaged F1-Score (Positive and Negative Classes): 83.57%
   Micro-Averaged F1-Score (All Classes): 75.5694%


.. comment: SST (So-Cal)

   General Statistics:
                precision    recall  f1-score   support

   positive       0.64      0.74      0.68       589
   negative       0.60      0.63      0.61       606
   neutral        0.22      0.11      0.14       254

   accuracy                            0.58      1449
   macro avg       0.48      0.49      0.48      1449
   weighted avg    0.55      0.58      0.56      1449

   Macro-Averaged F1-Score (Positive and Negative Classes): 64.83%
   Micro-Averaged F1-Score (All Classes): 58.2471%

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
|                                                                   So-Cal                                                                        |
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |  0.64  | 0.74 |      0.68     |  0.6   | 0.63 |      0.61     |  0.22  | 0.11 |     0.14    |      0.6483       |      0.5825       |
|                                                                   XLNET                                                                         |
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |  0.84  |  0.9 |      0.87     |  0.84  | 0.77 |      0.81     |  0.38  | 0.39 |     0.38    |      0.8357       |      0.7557       |
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

.. comment: IMDB

.. comment: SST

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |        |      |               |        |      |               |        |      |             |                   |                   |
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

.. comment: IMDB

.. comment: SST

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |        |      |               |        |      |               |        |      |             |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

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

.. comment: IMDB

.. comment: SST

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |        |      |               |        |      |               |        |      |             |                   |                   |
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

.. comment: IMDB

.. comment: SST

+-----------+-------------------------------+-------------------------------+-----------------------------+-------------------+-------------------+
| **Data**  |          **Positive**         |           **Negative**        |          **Neutral**        | :math:`Macro F_1` | :math:`Micro F_1` |
+           +--------+------+---------------+--------+------+---------------+--------+------+-------------+                   +                   +
|           |    P   |   R  |  :math:`F_1`  |   P    |   R  |  :math:`F_1`  |    P   |   R  | :math:`F_1` |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+
| IMDB      |        |      |               |        |      |               |        |      |             |                   |                   |
| SST       |        |      |               |        |      |               |        |      |             |                   |                   |
+-----------+--------+------+---------------+--------+------+---------------+--------+------+-------------+-------------------+-------------------+

.. _IMDB: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
.. _Stanford Sentiment Treebank: http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip


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
