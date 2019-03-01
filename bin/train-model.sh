#!/usr/bin/env bash
###############################################################################
###############################################################################
##                                                                           ##
## train-model.sh                                                            ##
## Trains financial news sentiment analysis classifiers.                     ##
##                                                                           ##
## Usage:                                                                    ##
##                                                                           ##
##  train-model.sh                                                           ##
##                                                                           ##
###############################################################################
###############################################################################
source activate fsa
export PYTHONPATH=/opt/code/github/Financial-News-Sentiment-Analysis/python
cd ../util
python train-classifier.py --classifier nb
python train-classifier.py --classifier nb --include-words

python train-classifier.py --classifier nbsvm
python train-classifier.py --classifier nbsvm --include-words


