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
import re

from util.text import validate_label

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def concatenate_trans(save_dir, dataset_csv):

    dataset_csv = os.path.join(save_dir, dataset_csv)

    f1 = open(dataset_csv, "rb")

    csvReader_dataset = csv.reader(f1)

    segments = []
    # row_num =1

    for row in csvReader_dataset:
        # print (row_num)
        # print (row)
        wav_path = os.path.join(save_dir, row[2]+".wav")
        wav_size = os.path.getsize(wav_path)
        transcript = expand_abbreviations(row[4])
        transcript = re.sub('[^a-zA-Z \n\.]|\.', '', transcript)
        transcript = unicodedata.normalize("NFKD", unicode(transcript)) \
            .encode("ascii", "ignore") \
            .decode("ascii", "ignore")
        transcript = transcript.lower().strip()
        segments.append([wav_path.replace("jonghyuk", "chico2121"), wav_size, transcript])
        # row_num +=1

    dataset = pandas.DataFrame(data=segments, columns=["wav_filename", "wav_filesize", "transcript"])
    dataset.to_csv(os.path.join(save_dir, dataset_csv.split('.')[0]+"_ds.csv"), index=False, encoding='utf-8')


if __name__ == "__main__":
    concatenate_trans("/home/jonghyuk/pycharmProjects/DeepSpeech/data/lj/", "all_te.csv")
    concatenate_trans("/home/jonghyuk/pycharmProjects/DeepSpeech/data/lj/", "sup_per_2.csv")
    concatenate_trans("/home/jonghyuk/pycharmProjects/DeepSpeech/data/lj/", "sup_per_2_tr.csv")
    concatenate_trans("/home/jonghyuk/pycharmProjects/DeepSpeech/data/lj/", "sup_per_2_va.csv")