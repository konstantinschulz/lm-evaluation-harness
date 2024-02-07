#!/bin/bash

MODEL_CHECKPOINTS="./checkpoints.txt"
TASKS="lambada_openai,arc_easy"
OUTPUT_PATH="./out"
BATCH_SIZE=32
CHECKPOINT_ITER=53100

ARRAY_RANGE="0-$(("$(wc -l "$MODEL_CHECKPOINTS")" - 1))"

sbatch --verbose --time=10:00:00 --array=$ARRAY_RNG\
    run_megatron_server_client.sbatch \
    $MODEL_CHECKPOINTS \
    $TASKS \
    $OUTPUT_PATH \
    $BATCH_SIZE \
    $CHECKPOINT_ITER
