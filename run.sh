#!/bin/bash

EXPERIMENT_NAME=testrandnn

PYTHONPATH=. python3 experiment/run_experiment.py \
    --experiment-config exp-config.yaml \
    --benchmarks libpng-1.2.56 \
    --experiment-name $EXPERIMENT_NAME \
    --fuzzers afl \
    -a
