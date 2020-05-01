#!/usr/bin/env python
"""
Quickly plot a dataset
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from argparse import Namespace
from emma.attacks.leakagemodels import LeakageModel, LeakageModelType
from emma.processing.dsp import *
from emma.io.io import get_trace_set
from irradiant.utils import disable_mpl_logging
from common import *

disable_mpl_logging()
max_samples = int(56000000/2)  # 0.5 secs @ 56 Msps
filter_method = 'abs'


def plot_dataset(args):
    listing = sorted(list(os.listdir(args.dataset_dir)))
    traces_gathered = 0
    print("Gathering traces")
    for i, entry in enumerate(listing):
        print("\r%d / %d                            " % (i // 4, len(listing) // 4), end='')
        if '_traces.npy' in entry:
            trace_name = entry.rpartition('_traces.npy')[0]
            trace_path = os.path.join(args.dataset_dir, entry)
            meta_path = os.path.join(args.dataset_dir, trace_name + "_meta.p")

            trace_set = get_trace_set(trace_path, 'cw', remote=False)
            meta_trace_set = load_meta(meta_path)

            for i, trace in enumerate(trace_set.traces):
                signal = trace.signal
                meta = meta_trace_set[i]
                if meta["datatype"] == "complex64":
                    signal = np.abs(signal)

                if args.filter:
                    signal = filter_trace(signal, filter_method)

                plt.title(meta["op"])
                if len(signal) > max_samples:
                    print("Splitting signal because it is too large")
                    for j in range(0, len(signal), max_samples):
                        plt.plot(signal[j:j+max_samples])
                        plt.show()
                else:
                    plt.plot(signal)
                    plot_meta(meta)

                traces_gathered += 1
                if traces_gathered >= args.num_traces:
                    return
                if args.separate:
                    plt.show()
                if args.specgram:
                    debug_trace_specgram(signal, 56000000, title=meta["op"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quickly plot a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to root directory of dataset.")
    parser.add_argument("num_traces", type=int, default=1, help="Number of traces to plot")
    parser.add_argument("--filter", default=False, action="store_true", help="Prefilter traces with a low pass filter for easier visual inspection.")
    parser.add_argument("--separate", default=False, action="store_true", help="Show plots separately.")
    parser.add_argument("--specgram", default=False, action="store_true", help="Also plot spectogram.")
    args = parser.parse_args()

    #lm = LeakageModel(Namespace(leakage_model=LeakageModelType.HMAC_HW, key_low=key_low, key_high=key_high))
    plot_dataset(args)
    if not args.separate:
        plt.show()
