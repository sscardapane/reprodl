#!/bin/sh
isort --profile black .
black .
flake8 . --count --select=B9