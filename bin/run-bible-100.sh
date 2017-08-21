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

python -u DeepSpeech.py \
  --train_files "$COMPUTE_DATA_DIR/bible-train-100-d.csv" \
  --dev_files "$COMPUTE_DATA_DIR/bible-dev-100-d.csv" \
  --test_files "$COMPUTE_DATA_DIR/bible-test-100-d.csv" \
  --train_batch_size 1 \
  --dev_batch_size 1 \
  --test_batch_size 1 \
  --learning_rate 0.0001 \
  --epoch=20 \
  --display_step 1 \
  --validation_step 5 \
  --checkpoint_dir "testcps/bible" \
  "$@"
