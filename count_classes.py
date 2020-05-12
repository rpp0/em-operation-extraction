#!/usr/bin/env python

import argparse
import os
from common import *
from collections import defaultdict

parser = argparse.ArgumentParser(description="Count number of classes in dataset.")
parser.add_argument("dataset_name", type=str, help="Name of the dataset.")
parser.add_argument("--datasets-root", type=str, default="./datasets/", help="Root of all datasets.")
args = parser.parse_args()

dataset_path = os.path.join(args.datasets_root, args.dataset_name)
dataset_files = list(os.listdir(dataset_path))
count_dict = defaultdict(lambda: 0)

for dataset_file in dataset_files:
    if '_traces.npy' in dataset_file:
        # Get metadata and trace paths
        trace_name = dataset_file.rpartition('_traces.npy')[0]
        meta_name = trace_name + "_meta.p"
        meta_path = os.path.join(dataset_path, meta_name)

        # Get trace metadata
        meta_trace_set = load_meta(meta_path)

        for meta in meta_trace_set:
            count_dict[meta["op"]] += 1

for k, v in count_dict.items():
    print("%s: %d" % (k, v))
