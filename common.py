import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import matplotlib.colors
import cv2
from emma.processing.dsp import *
from emma.io.io import get_trace_set
from matplotlib import collections as mc


op_to_int = {
    "aes": 0,
    "sha1prf": 1,
    "hmacsha1": 2,
    "des_openssl": 3,
    "aes_openssl": 4,
    "aes_tiny": 5,
    "sha1": 6,
    "sha1transform": 7,
    "noise": 8,
}

int_to_op = {
    0: "aes",
    1: "sha1prf",
    2: "hmacsha1",
    3: "des_openssl",
    4: "aes_openssl",
    5: "aes_tiny",
    6: "sha1",
    7: "sha1transform",
    8: "noise",
}


def snippetize(trace, snippet_length=131072, reference_length=10000000):
    """
    Cut long trace into snippets of length snippet_length to obtain trace around length reference_length
    :param trace:
    :return:
    """
    trace_len = len(trace)
    num_snippets = int(reference_length / snippet_length)
    snippet_step = int(trace_len / num_snippets)
    result = np.zeros(num_snippets * snippet_length)

    result = []
    for snippet_index, i in enumerate(range(0, len(trace), snippet_step)):
        if len(trace[i:i+snippet_length]) == snippet_length:
            result.append(trace[i:i+snippet_length])

    return np.array(result).flatten()


def get_stft(trace, sample_rate=56000000, show_plot=True, overlap_rate=0.90, fft_size=512):
    overlap = int(fft_size * overlap_rate)
    f, t, Sxx = scipy.signal.spectrogram(trace, fs=sample_rate, window=('tukey', 0.25), nperseg=fft_size, noverlap=overlap, nfft=fft_size, detrend=False, return_onesided=False, scaling='density', axis=-1, mode='psd')
    n = matplotlib.colors.LogNorm(vmin=Sxx.min()+0.0000000000001, vmax=Sxx.max(), clip=False)
    if show_plot:
        plt.imshow(Sxx, norm=n)
        plt.show()

    return Sxx


def plot_stft(Sxx, meta=None, norm=True, show=True):
    if norm:
        n = matplotlib.colors.LogNorm(vmin=Sxx.min()+0.0000000000001, vmax=Sxx.max(), clip=False)
        plt.imshow(Sxx, norm=n)
    else:
        plt.imshow(Sxx)
    if meta is not None:
        plot_meta(meta, extent=512)
    if show:
        plt.show()


def opencv_correlate(trace, template, window_limit=None, show_plot=True, method="norm_corr"):
    h, w = template.shape
    if method == "norm_corr":
        res = cv2.matchTemplate(trace[:, 0:window_limit], template, cv2.TM_CCORR_NORMED)
    else:
        res = cv2.matchTemplate(trace[:, 0:window_limit], template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method == "norm_corr":
        top_left = max_loc
    else:
        top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if show_plot:
        trace[:, top_left[0]-1] = 1 / 1e8
        trace[:, top_left[0]] = 1/1e8
        trace[:, top_left[0]+1] = 1 / 1e8
        trace[:, bottom_right[0]-1] = 1 / 1e8
        trace[:, bottom_right[0]] = 1/1e8
        trace[:, bottom_right[0]+1] = 1 / 1e8

    """
    n = matplotlib.colors.LogNorm(vmin=res.min(), vmax=res.max(), clip=False)
    plt.imshow(np.array([res[0]]*500), norm=n)
    plt.show()
    """

    if show_plot:
        n = matplotlib.colors.LogNorm(vmin=trace.min(), vmax=trace.max(), clip=False)
        plt.imshow(trace, norm=n)
        plt.show()
    return res, top_left[0]


def get_onehot(op_name):
    a = np.zeros(len(op_to_int))
    a[op_to_int[op_name]] = 1
    return a


def squared_diff(trace, template):
    template_len = len(template)
    len_diff = len(trace) - template_len
    if len_diff < 0:
        raise Exception("Template must be longer than trace")

    #trace = trace - np.mean(trace)
    #template = template - np.mean(template)

    results = np.zeros(len_diff + 1)
    best_result = np.inf
    best_index = 0
    for i in range(0, len_diff + 1):
        section = trace[i:i+template_len]
        result = np.sum(np.square(np.subtract(section, template))) / template_len
        results[i] = result
        if result < best_result:
            best_result = result
            best_index = i

    return best_index, best_result


def debug_trace_specgram(trace, fs, nfft=512, overlap=0.70, title=""):
    noverlap = nfft * overlap
    plt.specgram(trace, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap='plasma', mode='psd', scale='dB')
    plt.tight_layout()
    plt.title(title)
    plt.show()


def filter_trace(trace, filter_method):
    #debug_trace_specgram(trace, sample_rate)

    if filter_method == 'none':
        return trace
    elif filter_method == 'ifreq':
        filtered_trace = ifreq(trace)
    elif filter_method == 'abs_nofilt':
        filtered_trace = np.abs(trace)
    elif filter_method == 'abs':
        #debug_trace_specgram(trace, sample_rate)
        filtered_trace = butter_filter(trace, 1, 0.001, 'high', None)  # Remove low freqs
        #debug_trace_specgram(filtered_trace, sample_rate)
        filtered_trace = np.abs(filtered_trace)
        filtered_trace = butter_filter(filtered_trace, 1, 0.001, 'high', None)  # Remove low freqs
    elif filter_method == 'complex_hp':
        filtered_trace = butter_filter(trace, 1, 0.001, 'high', None)  # Remove low freqs
    else:
        raise Exception("unknown method")

    return filtered_trace


def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        meta_trace_set = pickle.load(f)
    return meta_trace_set


def save_meta(meta_trace_set, meta_path):
    with open(meta_path, "wb") as f:
        pickle.dump(meta_trace_set, f)


def plot_meta(meta, extent=1):
    if "left_bound" in meta and "right_bound" in meta:
        l = meta["left_bound"]
        r = meta["right_bound"]  # Non-inclusive
        lc = mc.LineCollection([[(l, -extent), (l, extent)], [(r-1, -extent), (r-1, extent)]], linewidths=2, color="red")
        plt.gca().add_collection(lc)


def get_arch_signal_paths(subdir=""):
    listing = list(os.listdir('./arch%s' % subdir))
    result = []
    for entry in listing:
        if '.npy' in entry:
            result.append(entry)
    return result


class ConfusionMatrix:
    def __init__(self, name):
        self.matrix = np.zeros((len(op_to_int), len(op_to_int)))
        self.name = name
        self.results_dir = "./results/"

    def save(self):
        print("Saving confusion matrix %s" % self.name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(self.results_dir, "cm-%s.p" % self.name), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name):
        with open(os.path.join(self.results_dir, "cm-%s.p" % name), "rb") as f:
            return pickle.load(f)

    def add(self, true_op, predicted_op):
        true_int = None
        predicted_int = None

        if type(true_op) is str:
            try:
                true_int = op_to_int[true_op]
            except KeyError:
                print("Could not get op_to_int key for op %s" % true_op)
                exit(1)
        elif type(true_op) is int:
            true_int = true_op
        else:
            print(type(true_op))
            print(true_op)
            raise ValueError

        if type(predicted_op) is str:
            try:
                predicted_int = op_to_int[predicted_op]
            except KeyError:
                print("Could not get op_to_int key for op %s" % predicted_op)
                exit(1)
        elif type(predicted_op) is int:
            predicted_int = predicted_op
        else:
            print(type(predicted_op))
            print(predicted_op)
            raise ValueError

        self.matrix[true_int, predicted_int] += 1

    def add_onehot(self, true_label, predicted_label):
        true_int = int(np.argmax(true_label))
        predicted_int = int(np.argmax(predicted_label))

        self.add(true_int, predicted_int)

    def print(self):
        temp = np.get_printoptions()
        np.set_printoptions(threshold=np.inf)
        print(self.matrix)
        np.set_printoptions(**temp)


class DistanceCounter:
    def __init__(self, name, num_classes, input_size):
        self.name = name
        self.num_classes = num_classes
        self.input_size = input_size
        self.noise_class = num_classes - 1
        self.results_dir = "./results/"
        self.num_nonnoise_traces = 0
        self.total_distance_squared = 0.0
        self.total_distance_abs = 0.0

    def save(self):
        print("Saving distance counter %s" % self.name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(self.results_dir, "dc-%s.p" % self.name), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name):
        with open(os.path.join(self.results_dir, "dc-%s.p" % name), "rb") as f:
            return pickle.load(f)

    def add_value(self, true_label, predicted_label):
        if true_label[self.noise_class] == 1:  # Skip distances for 'noise' class
            return

        self.num_nonnoise_traces += 1

        pred_left = predicted_label[self.num_classes]
        pred_right = predicted_label[self.num_classes+1]
        true_left = true_label[self.num_classes]
        true_right = true_label[self.num_classes+1]

        self.total_distance_squared += np.square((pred_left*self.input_size) - (true_left*self.input_size))
        self.total_distance_squared += np.square((pred_right*self.input_size) - (true_right*self.input_size))
        self.total_distance_abs += np.abs((pred_left*self.input_size) - (true_left*self.input_size))
        self.total_distance_abs += np.abs((pred_right * self.input_size) - (true_right * self.input_size))

    def print(self):
        print("Distance squared error totaled %f for %d valid traces. This is %f samples squared per trace on average." %
              (self.total_distance_squared, self.num_nonnoise_traces, self.total_distance_squared / self.num_nonnoise_traces))
        print("Distance absolute error totaled %f for %d valid traces. This is %f samples (%f microsecs) per trace on average." %
              (self.total_distance_abs, self.num_nonnoise_traces, self.total_distance_abs / self.num_nonnoise_traces, self.total_distance_abs / self.num_nonnoise_traces / 56000000 * 1000000))


def get_tracesets_and_meta(dataset_path):
    dataset_files = list(os.listdir(dataset_path))

    for i, dataset_file in enumerate(dataset_files):
        if '_traces.npy' in dataset_file:
            # Get metadata and trace paths
            trace_name = dataset_file.rpartition('_traces.npy')[0]
            meta_name = trace_name + "_meta.p"
            trace_path = os.path.join(dataset_path, dataset_file)
            meta_path = os.path.join(dataset_path, meta_name)

            # Get traces
            trace_set = get_trace_set(trace_path, 'cw', remote=False)

            # Get trace metadata
            meta_trace_set = load_meta(meta_path)
            yield trace_set, meta_trace_set, meta_path
