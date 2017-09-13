#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="data/lj"
fi;

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/sup_per_2_tr_ds.csv" ]; then
    echo "Warning: It looks like you don't have the LJ corpus"            \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the Bible data is located"
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/lj"))')
fi

CUDA_VISIBLE_DEVICES=0 python -u DeepSpeech.py \
  --train_files "$COMPUTE_DATA_DIR/sup_per_2_tr_ds.csv" \
  --dev_files "$COMPUTE_DATA_DIR/sup_per_2_va_ds.csv" \
  --test_files "$COMPUTE_DATA_DIR/all_te_ds.csv" \
  --n_hidden 1024 \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --learning_rate 0.0001 \
  --epoch=1000 \
  --display_step 1 \
  --validation_step 3 \
  --summary_secs 10 \
  --checkpoint_dir "cps/lj" \
  --export_dir "exp/lj" \
  --max_to_keep None \
  --early_stop False \
  "$@"
