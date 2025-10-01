#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p "$SCRIPT_DIR/raw"
curl -L https://osf.io/download/w3gan/ -o "$SCRIPT_DIR/raw/fixations-corrected.csv"
curl -L https://osf.io/download/vgp9a/ -o "$SCRIPT_DIR/raw/stimuli.csv"

mkdir -p "$SCRIPT_DIR/fixations"
python "$SCRIPT_DIR/extract.py"
