#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import fnmatch
import pandas
import subprocess
import unicodedata
import wave
import codecs
import csv

from util.text import validate_label


def _download_and_preprocess_data(data_dir):
    data_dir = os.path.join(data_dir, "bible")
    files = []

    for root, dirnames, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, "*.wav"):
            file_path = os.path.join(root, filename)
            wav_size = os.path.getsize(file_path)
            files.append((file_path, wav_size))

    dataset = pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize"])
    dataset.to_csv(os.path.join(data_dir, "test.csv"), index=False, encoding='utf-8')

def concatenate_trans(save_dir, trans, dataset_csv):
    f1 = codecs.open(trans, "rb", encoding="utf-8")
    f2 = open(dataset_csv, "r")
    csvReader_trans = csv.reader(f1)
    csvReader_dataset = csv.reader(f2)

    rows_trans = []
    rows_dataset = []
    for row in csvReader_trans:
        rows_trans.append(row)
    for row in csvReader_dataset:
        rows_dataset.append(row)

    for row_trans in rows_trans:
        if row_trans[1] == "":
            continue
        transcript = unicodedata.normalize("NFKD", unicode(row_trans[1])) \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore")
        transcript = transcript.lower().strip()

        for row_dataset in rows_dataset:
            if row_trans[0] + ".wav" in row_dataset[0]:
                row_dataset.append(transcript)

    del rows_dataset[0]

    for row_dataset in rows_dataset:
        if len(row_dataset) == 4:
            del row_dataset[3]

    dataset = pandas.DataFrame(data=rows_dataset, columns=["wav_filename", "wav_filesize", "transcript"])
    dataset.to_csv(os.path.join(save_dir, "bible.csv"), index=False, encoding='utf-8')

def _split_sets(bibledata_path, csvfile):
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    csvfilepath = bibledata_path + csvfile
    df = pandas.read_csv(csvfilepath)


    train_beg = 0
    train_end = int(0.8 * len(df))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(df)

    train_files, dev_files, test_files = df[train_beg:train_end], df[dev_beg:dev_end], df[test_beg:test_end]

    # # Write sets to disk as CSV files
    train_files.to_csv(os.path.join(bibledata_path, "bible-train.csv"), index=False)
    dev_files.to_csv(os.path.join(bibledata_path, "bible-dev.csv"), index=False)
    test_files.to_csv(os.path.join(bibledata_path, "bible-test.csv"), index=False)


def _delete_special_characters(bibledata_path, csvfile):

    csvfilepath = bibledata_path + csvfile
    df = pandas.read_csv(csvfilepath)
    new_trans = df["transcript"].str.replace('[^a-zA-Z0-9 \n\.]|\.', '')
    new_df = df[["wav_filename", "wav_filesize"]]
    new_df["transcript"] = new_trans

    new_df.to_csv(os.path.join(bibledata_path, "bible_delete_special_characters.csv"), index=False, encoding='utf-8')


if __name__ == "__main__":
    # _download_and_preprocess_data("/home/jonghyuk/pycharmProjects/DeepSpeech/data")
    # concatenate_trans("/home/jonghyuk/pycharmProjects/DeepSpeech/data/bible/", "/home/jonghyuk/pycharmProjects/DeepSpeech/data/text.csv", "/home/jonghyuk/pycharmProjects/DeepSpeech/data/bible/test.csv")
    # _delete_special_characters("/home/jonghyuk/pycharmProjects/DeepSpeech/data/bible/", "bible.csv")
    _split_sets("/home/jonghyuk/pycharmProjects/DeepSpeech/data/bible/", "bible_delete_special_characters.csv")
