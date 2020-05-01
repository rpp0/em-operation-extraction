#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
sys.path.append("./lib/fast-wavenet")
import logging
import sklearn.preprocessing
import argparse
from emma.emcap.sdr import SDR
from emma.emcap.types import *
from emma.utils.configargumentparser import ConfigArgumentParser
from emma.processing.dsp import *
from emma.emcap.protocol import Protocol
from emma.emcap.inputgenerator import InputGenerator
from common import *
from wavenet.models import BestCNNBB2D, BestCNN2D, get_cnnbb_loss
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
for l in ('matplotlib', 'matplotlib.font_manager'):
    mpl_logger = logging.getLogger(l)
    mpl_logger.setLevel(logging.WARNING)

num_classes = 9
num_stft_segments = 512
input_size = (num_stft_segments, num_stft_segments)
window_size = 131072+256


def preprocess(trace):
    trace = pad_to_length(trace[20000:], window_size)
    trace = get_stft(trace, overlap_rate=0.5, show_plot=False)
    trace = sklearn.preprocessing.normalize(trace)
    return trace


def plot_prediction(trace, prediction, num, has_meta=False):
    print(prediction)

    predicted_class = np.argmax(prediction[0:num_classes])

    fake_meta = None
    if has_meta:
        mid = prediction[num_classes] * num_stft_segments
        dist = prediction[num_classes + 1] * num_stft_segments
        fake_left = mid - (dist / 2.0)
        fake_right = fake_left + dist
        fake_meta = {"left_bound": fake_left, "right_bound": fake_right}

    plt.title("Predicted class: %s" % int_to_op[predicted_class])
    plot_stft(trace, meta=fake_meta, show=False)
    with PdfPages('/tmp/%d.pdf' % num) as pdf:
        pdf.savefig()
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real time demo of the classification.")
    parser.add_argument("--bounding-boxes", default=False, action="store_true", help="Also predict bounding boxes.")
    args = parser.parse_args()
    sdr_args = {'hw': 'usrp', 'samp_rate': 56000000, 'freq': 240e6, 'gain': 25, 'ds_mode': False, 'agc': False,
                'otw_format': "sc8"}
    sdr = SDR(**sdr_args)
    bounding_boxes = args.bounding_boxes

    if bounding_boxes:
        model = BestCNNBB2D(input_size, num_classes, load=False)
        model.model.load_weights("/tmp/best_cnn_bb_2d.h5")
    else:
        model = BestCNN2D(input_size, num_classes, load=False)
        model.model.load_weights("/tmp/best_cnn_2d.h5")

    # SHA1-PRF specific
    input_generator = InputGenerator("sha1prf_uniform_pmk_uniform_data")
    protocol = Protocol(ProtocolType.NODEMCU, ProtocolVersion.V2)  # Setup communication protocol with target
    protocol.start_session("/dev/ttyUSB0")

    traces = []
    for i in range(256):
        pmk, data = input_generator.generate_key_and_plaintext()
        protocol.request_sha1_prf(data, pmk)

        sdr.start_capture()
        protocol.ack_capture_started()
        protocol.wait_operation_done()
        trace = sdr.stop_capture()

        trace = preprocess(trace)
        print(trace.shape)
        batch = np.array([trace, np.zeros(11 if bounding_boxes else 9)]).reshape((1, 2))
        print(batch.shape)

        # Feed
        prediction = model.test_batch(batch)[0]
        plot_prediction(trace, prediction, i, has_meta=bounding_boxes)

