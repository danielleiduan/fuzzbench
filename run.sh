#!/bin/bash

EXPERIMENT_NAME=testrandnn

PYTHONPATH=. python3 experiment/run_experiment.py \
    --experiment-config exp-config.yaml \
    --benchmarks freetype2-2017 bloaty_fuzz_target \
    --experiment-name $EXPERIMENT_NAME \
    --fuzzers afl libfuzzer randnn
