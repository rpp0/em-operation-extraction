import numpy as np

from scipy.io import wavfile


def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]
    return get_wavenet_data(data)


def get_wavenet_data(data, resolution=256):
    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, resolution)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
    return inputs, targets

def get_normalized_data(data):
    # A bit of a hack, sorry no time
    data_ = normalize(data)
    return data_[0:-1], data_[1:]
