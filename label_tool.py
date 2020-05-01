#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Button
from common import *


def load_arch_signals(arch_type="notrigger"):
    result = {}
    for arch_path in get_arch_signal_paths(arch_type):
        if arch_type in arch_path:  # Use notrigger arch signals by default
            arch_name = arch_path.partition(".npy")[0].partition("-")[2].partition(arch_type)[0].rpartition("-")[0].rpartition("-")[0].replace("-", "_")
            if arch_name != '':
                result[arch_name] = np.load(os.path.join("./arch%s" % arch_type, arch_path))
    return result


use_stft = False  # Use OpenCV STFT matching as matching technique instead of correlation of AM demodulated signals
stft_scaling_factor = 0
use_boot = False or use_stft

if use_boot:
    arch_signals = load_arch_signals("boot")
else:
    arch_signals = load_arch_signals("notrigger")


def plot_lines(l, r):
    global lc
    lc.set_paths([[(l, -1), (l, 1)], [(r, -1), (r, 1)]])
    plt.draw()
    #plt.gca().draw_artist(lc)


def perform_pick(left_bound):
    global trace_index
    global metas

    meta = metas[trace_index]
    right_bound = left_bound + len(arch_signals[meta["op"]])
    metas[trace_index]["left_bound"] = left_bound
    metas[trace_index]["right_bound"] = right_bound
    print(metas[trace_index])
    plot_lines(left_bound, right_bound)


def perform_right_pick(right_bound):
    global trace_index
    global metas

    metas[trace_index]["right_bound"] = right_bound
    print(metas[trace_index])
    plot_lines(metas[trace_index]["left_bound"], right_bound)


def on_pick(event):
    line = event.artist
    xdata, ydata = line.get_data()
    ind = event.ind

    # Find best corr near clicked point
    best_corr = -1
    selected_point = 0
    for i in ind:
        corr = ydata[i]
        if corr > best_corr:
            best_corr = corr
            selected_point = xdata[i]
    print('Labeling at ', selected_point)

    if event.mouseevent.button == 3:  # Right
        perform_right_pick(selected_point)
    else:
        perform_pick(selected_point)


def autopick(correlations, offset=1000, window=14000, is_stft=False):
    if is_stft:
        global stft_scaling_factor
        left_bound = np.argmax(correlations)
        left_bound = int(stft_scaling_factor * left_bound)
        perform_pick(left_bound + offset)
    else:
        left_bound = np.argmax(correlations[offset:offset+window])
        perform_pick(left_bound+offset)


def noise_button(event):
    print("Setting 'bad' flag")
    global trace_index
    global metas
    metas[trace_index]["bad"] = True
    plt.close()


def next_button(event):
    plt.close()


def on_key(event):
    if event.key == 'n':
        noise_button(event)
    elif event.key == 'm':
        next_button(event)


def label_dataset(dataset_name, relabel=False):
    window = None
    global trace_index
    global metas
    global lc
    traceset_index = 0
    for traceset, metas, meta_path in get_tracesets_and_meta(dataset_name):
        traceset_index += 1
        for trace_index, trace in enumerate(traceset.traces):
            print("%d/%d (trace set %d)" % (trace_index, len(traceset.traces), traceset_index))
            meta = metas[trace_index]

            if meta["op"] == "noise":
                continue

            # Check if bounds are there and if so, draw them if relabel == True.
            if "left_bound" in meta and "right_bound" in meta:
                if relabel:
                    l = meta["left_bound"]
                    r = meta["right_bound"]
                    lc = mc.LineCollection([[(l, -1), (l, 1)], [(r, -1), (r, 1)]], linewidths=2, color="black")
                else:
                    continue
            else:
                lc = mc.LineCollection([], linewidths=2, color="black")
            f, (line_axis, spec_axis) = plt.subplots(2, 1)
            line_axis.add_collection(lc)

            # Filter the trace and correlate it
            filtered_trace = trace.signal[0:window]
            spec_axis.specgram(filtered_trace, NFFT=256, Fs=56000000, noverlap=128, cmap='plasma', mode='psd', scale='dB')
            if use_stft:
                global stft_scaling_factor
                stft_size = 512
                overlap = 0.90
                added_per_stft = stft_size - int(stft_size * overlap)
                stft_trace = get_stft(filtered_trace, show_plot=False)
                stft_size = 1+(len(filtered_trace)-stft_size) / added_per_stft
                stft_scaling_factor = len(filtered_trace) / stft_size
            filtered_trace = filter_trace(filtered_trace, 'abs')
            try:
                line_template = arch_signals[meta["op"]]
                if use_stft:
                    stft_template = get_stft(line_template, show_plot=False)
                if use_boot:
                    line_template = filter_trace(line_template, 'abs')
            except KeyError:
                print("No template for op %s. Setting template to impulse!" % meta["op"])
                line_template = [1]*len(filtered_trace)
                if use_stft:
                    stft_template = get_stft(line_template, show_plot=False)
                if use_boot:
                    line_template = filter_trace(line_template, 'abs')

            if use_stft:
                stft_correlations, _ = opencv_correlate(stft_trace, stft_template, show_plot=False)[0]
            line_correlations = fast_xcorr(filtered_trace, line_template, prefilter=False, required_corr=-1.0, normalized=True, debug=False, return_corr=False, return_corr_trace=True)
            if meta["op"] != "noise" and "left_bound" not in meta:
                if use_stft:
                    autopick(stft_correlations, is_stft=True)
                else:
                    autopick(line_correlations, is_stft=False)

            # Draw the plot and let user pick position
            plt.title(meta["op"])
            filtered_trace = butter_filter(filtered_trace, 1, 0.1, 'low', None)
            line_axis.plot(filtered_trace)
            line_axis.plot(line_correlations, picker=1)
            line_axis.set_xlim(0, len(filtered_trace))
            line_axis.set_ylim(-0.15, 0.15)
            plt.gcf().canvas.mpl_connect('pick_event', on_pick)
            plt.gcf().canvas.mpl_connect('key_press_event', on_key)

            button1 = plt.axes([0.7, 0.05, 0.1, 0.075])
            button2 = plt.axes([0.81, 0.05, 0.1, 0.075])
            bnoise = Button(button1, 'Noise')
            bnoise.on_clicked(noise_button)
            bnext = Button(button2, 'Next')
            bnext.on_clicked(next_button)

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

            plt.show()
            with open(meta_path, "wb") as f:
                pickle.dump(metas, f)


parser = argparse.ArgumentParser(description='Dataset labeler')
parser.add_argument("--datasets-root", type=str, default="./datasets/", help="Root of all datasets.")
parser.add_argument('--relabel', default=False, action='store_true', help='Relabel dataset')
args, unknown = parser.parse_known_args()

label_dataset(os.path.join(args.datasets_root, "nodemcu-random-label-val"), relabel=args.relabel)