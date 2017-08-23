#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="data/bible"
fi;

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/bible-train.csv" ]; then
    echo "Warning: It looks like you don't have the Bible corpus"            \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Bible data is located"
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/bible"))')
fi

CUDA_VISIBLE_DEVICES=0 python -u DeepSpeech.py \
  --train_files "$COMPUTE_DATA_DIR/bible-train.csv" \
  --dev_files "$COMPUTE_DATA_DIR/bible-dev.csv" \
  --test_files "$COMPUTE_DATA_DIR/bible-test.csv" \
  --n_hidden 1024 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --learning_rate 0.0001 \
  --epoch=10 \
  --display_step 1 \
  --validation_step 3 \
  --summary_secs 10 \
  --checkpoint_dir "cps/bible" \
  --export_dir "exp/bible"
  "$@"
