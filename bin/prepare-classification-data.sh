#!/usr/bin/env bash
###############################################################################
###############################################################################
##                                                                           ##
## prepare-classification-data.sh                                            ##
## Prepares financial news sentiment analysis classifier training and        ##
## evaluation data.                                                          ##
##                                                                           ##
## Usage:                                                                    ##
##                                                                           ##
##  prepare-classification-data.sh                                           ##
##                                                                           ##
###############################################################################
###############################################################################
source activate fsa
export PYTHONPATH=/opt/code/github/Financial-News-Sentiment-Analysis/python
cd ../util
python prepare-classification-data.py --dataset all-agree
python prepare-classification-data.py --dataset all-agree --include-words

python prepare-classification-data.py --dataset ad-hoc
python prepare-classification-data.py --dataset ad-hoc --include-words


