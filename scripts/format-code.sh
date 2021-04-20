#!/bin/sh
isort --profile black test_train.py train.py
black test_train.py train.py
flake8 test_train.py train.py --count --select=B9