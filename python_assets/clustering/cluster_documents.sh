#!/usr/bin/env bash

BASE_DIR=/home/acrem003/Documents/Cognac/repo/acrem003/onr/code/CDEC/python_assets/clustering
cd $BASE_DIR
OUTPUT=$(pipenv run python cluster_documents.py "$1" $2 $3)

echo $OUTPUT
