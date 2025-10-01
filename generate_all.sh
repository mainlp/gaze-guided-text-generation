#!/bin/bash
set -e

gaze_model=trf
beam_size=8
gpu=0

for gaze_weight in 0 2 -2; do
    python generate.py \
        --prompts stories/prompts.jsonl \
        --language-model meta-llama/Llama-3.2-3B-Instruct \
        --gaze-model $gaze_model \
        --gaze-weight $gaze_weight \
        --beam-size $beam_size \
        --gpu $gpu \
        > stories/output-Llama-3B/$gaze_model-gaze$gaze_weight-beam$beam_size.jsonl
done
