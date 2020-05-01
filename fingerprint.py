#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import traceback
from irradiant.utils import create_logger, Fore, disable_mpl_logging, StatusBar
disable_mpl_logging()
from emma.io.io import get_trace_set
from emma.processing.dsp import *
from emma.io.io import read_op_durations
from common import *

sample_rate = 56000000
debug = False
statusbar = StatusBar()


def debug_trace(trace, title=""):
    plot_trace = butter_filter(trace, 1, 0.02, 'low', None)
    plt.plot(plot_trace)
    plt.title(title)
    plt.show()


def find_trace(trace, template, return_indices=False):
    if len(trace) < len(template):
        return None

    #debug_trace(trace)

    index = fast_xcorr(trace[0:120000], template, normalized=True, prefilter=False, debug=debug)
    if index is None:
        return None

    if return_indices:  # Only return indices (don't cut trace)
        return index, index+len(template)
    else:  # Cut trace and return it
        trace = trace[index:index+len(template)]

        return trace


def load_trigger_template(op_duration):
    # DEPRECATED
    trigger = list(np.load("trigger.npy"))
    #return np.array(trigger + [1.0]*op_duration + trigger)
    return np.array(trigger)


def load_boot_template(target_name, op_name, sample_rate):
    boot_signal_name = "./archboot/%s-%s-%d-boot.npy" % (target_name, op_name, sample_rate)
    logger.info("Loading boot signal template %s" % boot_signal_name)
    return np.load(boot_signal_name)


def load_arch_template(target_name, op_name, sample_rate):
    arch_signal_name = "./arch/%s-%s-%d.npy" % (target_name, op_name, sample_rate)
    logger.info("Loading arch signal template %s" % arch_signal_name)
    return np.load(arch_signal_name)


def make_arch_signal(dataset, filter_method):
    logger.info("Making arch signal for dataset %s" % dataset)

    # Get target name and op name, durations and impulse to correlate with
    target_name, _, op_name = dataset.partition("-")
    template = load_boot_template(target_name, op_name, sample_rate)
    template = filter_trace(template, filter_method)

    # Get files to filter
    filtered_traces = []
    dataset_path = os.path.join(args.datasets_root, dataset)
    dataset_files = list(os.listdir(dataset_path))
    for i, dataset_file in enumerate(sorted(dataset_files)):
        if '_traces.npy' in dataset_file:
            statusbar.data["progress"] = "%d/%d" % (i // 4, len(dataset_files) // 4)
            trace_name = dataset_file.rpartition('_traces.npy')[0]
            path = os.path.join(args.datasets_root, dataset, dataset_file)
            trace_set = get_trace_set(path, 'cw', remote=False)

            batch_filtered_traces = []
            for trace in trace_set.traces:
                statusbar.data["batch"] = len(batch_filtered_traces)
                filtered_trace = filter_trace(trace.signal, filter_method)
                filtered_trace = find_trace(filtered_trace, template)
                if filtered_trace is not None:
                    batch_filtered_traces.append(filtered_trace)
            batch_mean_trace = np.mean(np.array(batch_filtered_traces), axis=0)
            filtered_traces.append(batch_mean_trace)

    arch_signal = np.mean(np.array(filtered_traces), axis=0)
    os.makedirs("./arch", exist_ok=True)
    np.save("./arch/%s-%s-%d.npy" % (target_name, op_name, sample_rate), arch_signal)


# TODO fix duplicate code
def make_stft_arch_signal(dataset, comparison_method):
    logger.info("Making STFT arch signal for dataset %s" % dataset)

    # Get target name and op name, durations and impulse to correlate with
    target_name, _, op_name = dataset.partition("-")
    template = load_boot_template(target_name, op_name, sample_rate)
    template_stft = get_stft(template, show_plot=False)

    # Get files to filter
    filtered_traces = []
    dataset_path = os.path.join(args.datasets_root, dataset)
    dataset_files = list(os.listdir(dataset_path))
    for i, dataset_file in enumerate(sorted(dataset_files)):
        if '_traces.npy' in dataset_file:
            statusbar.data["progress"] = "%d/%d" % (i // 4, len(dataset_files) // 4)
            trace_name = dataset_file.rpartition('_traces.npy')[0]
            path = os.path.join(args.datasets_root, dataset, dataset_file)
            trace_set = get_trace_set(path, 'cw', remote=False)

            batch_filtered_traces = []
            for trace in trace_set.traces:
                statusbar.data["batch"] = len(batch_filtered_traces)
                trace_stft = get_stft(pad_to_length(trace.signal, 131072), show_plot=False)
                try:
                    corr_trace, best_loc = opencv_correlate(trace_stft, template_stft, method=comparison_method, show_plot=False)
                except IndexError:
                    continue
                trace_stft = trace_stft[:, best_loc:best_loc+template_stft.shape[1]]
                if trace_stft.shape[1] < template_stft.shape[1]:
                    continue

                batch_filtered_traces.append(trace_stft)

            batch_mean_trace = np.mean(np.array(batch_filtered_traces), axis=0)
            template_stft = batch_mean_trace
            filtered_traces.append(batch_mean_trace)

            #n = matplotlib.colors.LogNorm(vmin=batch_mean_trace.min(), vmax=batch_mean_trace.max(), clip=False)
            #plt.imshow(batch_mean_trace, norm=n)
            #plt.show()

    arch_signal = np.mean(np.array(filtered_traces), axis=0)

    os.makedirs("./archstft", exist_ok=True)
    np.save("./archstft/%s-%s-%d-stft.npy" % (target_name, op_name, sample_rate), arch_signal)


def label_signal(dataset, filter_method):
    logger.info("Labeling dataset %s" % dataset)

    # Get target name and op name, durations and impulse to correlate with
    target_name, _, op_name = dataset.partition("-")
    op_durations = read_op_durations(target_name, sample_rate)
    template = load_arch_template(target_name, op_name, sample_rate)

    # Get files to label
    dataset_path = os.path.join(args.datasets_root, dataset)
    dataset_files = list(os.listdir(dataset_path))
    for i, dataset_file in enumerate(sorted(dataset_files)):
        if '_traces.npy' in dataset_file:
            statusbar.data["progress"] = "%d/%d" % (i // 4, len(dataset_files) // 4)

            # Get metadata and trace paths
            trace_name = dataset_file.rpartition('_traces.npy')[0]
            meta_name = trace_name + "_meta.p"
            trace_path = os.path.join(args.datasets_root, dataset, dataset_file)
            meta_path = os.path.join(args.datasets_root, dataset, meta_name)

            # Get traces
            trace_set = get_trace_set(trace_path, 'cw', remote=False)

            # Get trace metadata
            meta_trace_set = load_meta(meta_path)

            for j, trace in enumerate(trace_set.traces):
                statusbar.data["batch"] = j
                filtered_trace = filter_trace(trace.signal, filter_method)
                result = find_trace(filtered_trace, template, return_indices=True)
                if result is not None:
                    left, right = result
                    meta_trace_set[j]["left_bound"] = left + op_durations["trigger"]
                    meta_trace_set[j]["right_bound"] = right - op_durations["trigger"]

            save_meta(meta_trace_set, meta_path)


def make_boot_signal(dataset, filter_method):
    target_name, _, op_name = dataset.partition("-")
    confirm = input("Make boot signal for %s-%s? (y/N) " % (target_name, op_name))
    if confirm.lower() != "y":
        return

    dataset_path = os.path.join(args.datasets_root, dataset)
    dataset_files = list(os.listdir(dataset_path))
    for dataset_file in dataset_files:
        if '_traces.npy' in dataset_file:
            path = os.path.join(args.datasets_root, dataset, dataset_file)
            trace_set = get_trace_set(path, 'cw', remote=False)

            for trace in trace_set.traces:
                filtered_trace = filter_trace(trace.signal, filter_method)
                debug_trace(filtered_trace)
                start = input("Start or skip? ")
                if start == "skip":
                    continue
                end = input("End? ")
                os.makedirs("./archboot", exist_ok=True)
                np.save("./archboot/%s-%s-%d-boot.npy" % (target_name, op_name, sample_rate), trace.signal[int(start):int(end)])
                return


def experiment_roi_norm(test_dataset, filter_method, comparison_method, input_size=131072):
    # Load arch signals and apply the selected filtering method
    templates = {}
    subdir = "notrigger"
    for entry in get_arch_signal_paths(subdir):
        _, _, arch_signal_op = entry.partition("-")
        arch_signal_op = arch_signal_op.rpartition("-")[0].rpartition("-")[0].replace("-", "_")
        if arch_signal_op.strip() == '':
            continue
        templates[arch_signal_op] = np.load(os.path.join("./arch%s" % subdir, entry))
        print("Loaded arch signal for op: %s" % arch_signal_op)

    dataset_path = os.path.join(args.datasets_root, test_dataset)
    dataset_files = list(os.listdir(dataset_path))
    confusion_matrix = ConfusionMatrix("roi-norm-%s-%s" % (filter_method, comparison_method))

    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            statusbar.data["progress"] = "%d/%d" % (i // 4, len(dataset_files) // 4)

            # Get metadata and trace paths
            trace_name = dataset_file.rpartition('_traces.npy')[0]
            meta_name = trace_name + "_meta.p"
            trace_path = os.path.join(args.datasets_root, test_dataset, dataset_file)
            meta_path = os.path.join(args.datasets_root, test_dataset, meta_name)

            # Get traces
            trace_set = get_trace_set(trace_path, 'cw', remote=False)

            # Get trace metadata
            meta_trace_set = load_meta(meta_path)

            for j, trace in enumerate(trace_set.traces):
                try:
                    statusbar.data["batch"] = j
                    filtered_trace = pad_to_length(trace.signal, input_size)
                    filtered_trace = filter_trace(filtered_trace, filter_method)

                    # Find the template with the highest correlation
                    best_score = -1.0 if comparison_method == "norm_corr" else np.inf
                    best_op = None
                    for arch_signal_op, arch_signal in templates.items():
                        if comparison_method == "norm_corr":
                            _, corr = fast_xcorr(filtered_trace, arch_signal, normalized=True, prefilter=False, debug=False, return_corr=True)
                            if corr > best_score:
                                best_score = corr
                                best_op = arch_signal_op
                        elif comparison_method == "sq_diff":
                            _, diff = squared_diff(filtered_trace, arch_signal)
                            if diff < best_score:
                                best_score = diff
                                best_op = arch_signal_op
                        else:
                            raise Exception("Unknown comparison method %s" % comparison_method)

                    # Which is the correct op?
                    correct_op = meta_trace_set[j]["op"]
                    statusbar.data["predict"] = "%s -> %s                      " % (correct_op, best_op)
                    confusion_matrix.add(true_op=correct_op, predicted_op=best_op)
                except Exception:
                    traceback.print_exc()

    confusion_matrix.save()
    confusion_matrix.print()


def experiment_roi_stft(test_dataset, comparison_method, input_size=131072):
    # Load arch signals and apply the selected filtering method
    templates = {}
    subdir = "stftnotrigger"

    for entry in get_arch_signal_paths(subdir):
        _, _, arch_signal_op = entry.partition("-")
        arch_signal_op = arch_signal_op.rpartition("-")[0].rpartition("-")[0].replace("-", "_")
        if arch_signal_op.strip() == '':
            continue
        templates[arch_signal_op] = np.load(os.path.join("./arch%s" % subdir, entry))
        print("Loaded arch signal for op: %s" % arch_signal_op)

    dataset_path = os.path.join(args.datasets_root, test_dataset)
    dataset_files = list(os.listdir(dataset_path))
    confusion_matrix = ConfusionMatrix("roi-stft-%s" % comparison_method)

    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            statusbar.data["progress"] = "%d/%d" % (i // 4, len(dataset_files) // 4)

            # Get metadata and trace paths
            trace_name = dataset_file.rpartition('_traces.npy')[0]
            meta_name = trace_name + "_meta.p"
            trace_path = os.path.join(args.datasets_root, test_dataset, dataset_file)
            meta_path = os.path.join(args.datasets_root, test_dataset, meta_name)

            # Get traces
            trace_set = get_trace_set(trace_path, 'cw', remote=False)

            # Get trace metadata
            meta_trace_set = load_meta(meta_path)

            for j, trace in enumerate(trace_set.traces):
                try:
                    if meta_trace_set[j]["op"] == "noise":
                        continue
                    statusbar.data["batch"] = j
                    trace_stft = get_stft(pad_to_length(trace.signal, input_size), show_plot=False)

                    # Find the template with the highest correlation
                    best_score = -1.0 if comparison_method == "norm_corr" else np.inf
                    best_op = None
                    for arch_signal_op, arch_stft in templates.items():
                        corr_trace, best_loc = opencv_correlate(trace_stft, arch_stft, method=comparison_method, show_plot=False)
                        corr_trace = corr_trace[0]
                        score = max(corr_trace) if comparison_method == "norm_corr" else min(corr_trace)

                        if comparison_method == "norm_corr":
                            if score > best_score:
                                best_score = score
                                best_op = arch_signal_op
                        elif comparison_method == "sq_diff":
                            if score < best_score:
                                best_score = score
                                best_op = arch_signal_op
                        else:
                            raise Exception("Unknown comparison method %s" % comparison_method)

                    # Which is the correct op?
                    correct_op = meta_trace_set[j]["op"]
                    statusbar.data["predict"] = "%s -> %s                      " % (correct_op, best_op)
                    confusion_matrix.add(true_op=correct_op, predicted_op=best_op)
                except Exception:
                    traceback.print_exc()

    confusion_matrix.save()
    confusion_matrix.print()


def remove_arch_triggers():
    for arch_path in get_arch_signal_paths():
        if "notrigger" in arch_path:
            continue
        print(arch_path)
        arch_name = arch_path.partition(".npy")[0]
        new_arch_path = "%s-notrigger.npy" % arch_name
        signal = np.load(os.path.join("./arch/", arch_path))
        plt.plot(signal)
        plt.show()
        cut_signal = signal[1200:-1200]
        plt.plot(cut_signal)
        plt.show()

        os.makedirs("./archnotrigger", exist_ok=True)
        np.save(os.path.join("./archnotrigger/", new_arch_path), cut_signal)


def remove_arch_triggers_stft():
    for arch_path in get_arch_signal_paths("stft"):
        print(arch_path)
        arch_name = arch_path.partition(".npy")[0]
        new_arch_path = "%snotrigger.npy" % arch_name
        signal = np.load(os.path.join("./archstft/", arch_path))
        plot_stft(signal)
        cut = 17
        cut_signal = signal[:, cut:-cut]
        plot_stft(cut_signal)

        os.makedirs("./archstftnotrigger", exist_ok=True)
        np.save(os.path.join("./archstftnotrigger/", new_arch_path), cut_signal)


logger = create_logger("fingerprint", Fore.WHITE)
parser = argparse.ArgumentParser(description="Identify operations in traces of EM emissions.")
parser.add_argument("action", type=str, choices=["make_arch_signals", "make_stft_arch_signals", "make_boot_signals", "label_signals", "experiment_roi_norm", "experiment_roi_stft", "remove_trigger_from_arch", "remove_trigger_from_arch_stft"], help="Action to perform.")
parser.add_argument("filter_method", type=str, choices=["none", "ifreq", "abs_nofilt", "abs", "complex_hp"], help="Type of trace filtering to apply pre classification")
parser.add_argument("comparison_method", type=str, choices=["norm_corr", "sq_diff"], help="Type of comparison metric to use between templates and traces")
parser.add_argument("--datasets-root", type=str, default="./datasets/", help="Root of all datasets.")
args = parser.parse_args()

datasets = list(os.listdir(args.datasets_root))[::-1]  # SHA1-PRF first
#datasets = list(os.listdir(args.datasets_root))  # AES first
for ds in ["noise.npy", "nodemcu-fullconnect", "nodemcu-mix", "OLDnodemcu-random-train", "OLDnodemcu-random-test", "nodemcu-random-train", "nodemcu-random-test", "nodemcu-random-label-train", "nodemcu-random-label-test", "nodemcu-random-train2", "nodemcu-random-test2"]:
    try:
        datasets.remove(ds)
    except ValueError:
        print("Warning: didn't remove %s" % ds)
logger.info("Datasets: %s" % str(datasets))

if args.action == "make_arch_signals":
    for dataset in datasets:
        make_arch_signal(dataset, args.filter_method)
elif args.action == "make_stft_arch_signals":
    for dataset in datasets:
        make_stft_arch_signal(dataset, args.comparison_method)
elif args.action == "remove_trigger_from_arch":
    remove_arch_triggers()
elif args.action == "remove_trigger_from_arch_stft":
    remove_arch_triggers_stft()
elif args.action == "make_boot_signals":
    statusbar.detach()
    statusbar = None
    for dataset in datasets:
        make_boot_signal(dataset, args.filter_method)
elif args.action == "label_signals":
    for dataset in datasets:
        label_signal(dataset, args.filter_method)
elif args.action == "experiment_roi_norm":
    experiment_roi_norm("nodemcu-random-test2", args.filter_method, args.comparison_method)
elif args.action == "experiment_roi_stft":
    experiment_roi_stft("nodemcu-random-test2", args.comparison_method)
