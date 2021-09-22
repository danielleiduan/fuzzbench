#!/bin/bash

EXPERIMENT_NAME=testrandnn

PYTHONPATH=. python3 experiment/generate_report.py \
    experiments $EXPERIMENT_NAME \
    --quick \
    --fuzzers afl libfuzzer randnn \
    --report-dir ${EXPERIMENT_NAME}_report
