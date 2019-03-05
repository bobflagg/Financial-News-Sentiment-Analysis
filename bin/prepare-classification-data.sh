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
python prepare-classification-data.py --dataset all-agree --ftype very-strict &
python prepare-classification-data.py --dataset all-agree --ftype strict &
python prepare-classification-data.py --dataset all-agree --ftype regular &
python prepare-classification-data.py --dataset all-agree --ftype flush &

python prepare-classification-data.py --dataset ad-hoc --ftype very-strict &
python prepare-classification-data.py --dataset ad-hoc --ftype strict &
python prepare-classification-data.py --dataset ad-hoc --ftype regular &
python prepare-classification-data.py --dataset ad-hoc --ftype flush &

wait
echo "DONE PREPARING DATA!!!"

