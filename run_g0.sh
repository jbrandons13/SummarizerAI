#!/bin/bash
PYTHONPATH=. python -u pipeline/facet/runner.py > runner.log 2>&1
PYTHONPATH=. python -u pipeline/facet/scorer.py > scorer.log 2>&1
