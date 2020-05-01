#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.text as text
import matplotlib.patches as patches
import argparse
import sys
from matplotlib.backends.backend_pdf import PdfPages
from emma.processing.dsp import butter_filter
from common import *


def delete_border(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)


title_dict = {
    "des-openssl": "OpenSSL DES",
    "sha1prf": "SHA1-PRF",
    "sha1transform": "SHA1Transform",
    "aes-openssl": "OpenSSL AES",
    "hmacsha1": "HMAC-SHA1",
    "sha1": "SHA1",
    "aes": "Native AES",
    "aes-tiny": "TinyAES"
}

def generate_graph():
    # Setup
    plt.rcParams.update({
        'font.size': 14,
        'font.sans-serif': ["Linux Libertine"],
    })
    fs = 56.0e6
    bins_to_draw = 512
    vmin = 0.0000000000000001
    vmax = 0.000000001
    fft_size = 512
    overlap = 0.90
    colormap = 'plasma'
    samples_per_bin = fft_size - int(fft_size * overlap)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=300)

    for i, entry in enumerate(get_arch_signal_paths("stftnotrigger")):
        current_axis = axes[i // 3][i % 3]
        arch_signal_name = entry.rpartition('.npy')[0]
        path = os.path.join("./archstftnotrigger", entry)
        arch = np.load(path)
        arch = np.fft.fftshift(arch, axes=0)
        n = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
        current_axis.imshow(arch, norm=n, interpolation='bicubic', aspect='auto', cmap=colormap)

        # Fix y axis
        freqs = np.fft.fftshift(np.fft.fftfreq(512, d=1.0/fs))
        current_axis.set_yticks([0, 128, 256, 384, 512])
        freqs = list(freqs)
        freqs.append(fs/2)
        current_axis.set_yticklabels([int(freqs[x] / 1e6) for x in current_axis.get_yticks()])

        # Fix x axis
        print(arch.shape[1])
        current_axis.set_xlim(0, arch.shape[1] - 2)
        current_axis.set_xticks([x for x in range(0, (arch.shape[1] - 2), (arch.shape[1] - 2) // 5)])
        current_axis.set_xticklabels([f"{int(x*samples_per_bin):,d}" for x in current_axis.get_xticks()])
        current_axis.set_xlabel("Time (samples)")
        current_axis.set_ylabel("Frequency (MHz)")
        current_axis.set_title(title_dict[arch_signal_name.partition("-56")[0].partition("-")[2]])

    # Render
    axes[2][2].set_visible(False)
    plt.tight_layout()
    with PdfPages('allstftarch.pdf') as pdf:
        pdf.savefig()


parser = argparse.ArgumentParser(description="Draw all STFT arch signals.")
args = parser.parse_args()

generate_graph()
