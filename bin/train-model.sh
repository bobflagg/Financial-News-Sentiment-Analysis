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
python train-classifier.py --classifier nb --ftype very-strict --n-gram-bound 2 --use-tf-idf 
python train-classifier.py --classifier nb --ftype strict --n-gram-bound 2 --use-tf-idf 
python train-classifier.py --classifier nb --ftype regular --n-gram-bound 2 --use-tf-idf 
python train-classifier.py --classifier nb --ftype flush --n-gram-bound 2 --use-tf-idf 

python train-classifier.py --classifier nbsvm --ftype very-strict --n-gram-bound 2 --binary
python train-classifier.py --classifier nbsvm --ftype strict --n-gram-bound 2 --binary 
python train-classifier.py --classifier nbsvm --ftype regular --n-gram-bound 2 --binary 
python train-classifier.py --classifier nbsvm --ftype flush --n-gram-bound 2 --binary 

wait
echo "DONE TRAINING MODELS!!!"

