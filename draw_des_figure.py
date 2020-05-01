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


def print_des_indices(meta):
    for i, m in enumerate(meta):
        if m["op"] == "des_openssl":
            print(i)


def draw_box(axis, start, end, message):
    fontprop = matplotlib.font_manager.FontProperties(family="Linux Libertine", size=10)

    # Create a Rectangle patch
    offset = 20
    box = patches.Rectangle((start, offset), end-start, 512-(2*offset), linewidth=2, edgecolor='black', facecolor='none')
    axis.add_patch(box)

    description = text.Text(end - (end-start)//2, 90, message, ha='center', va='bottom', axes=axis, fontproperties=fontprop)
    axis.add_artist(description)


def generate_graph(spike_path, des_path, spike_index, des_index):
    # Setup
    plt.rcParams.update({
        'font.size': 14,
        'font.sans-serif': ["Linux Libertine"],
    })
    fs = 56.0e6
    bins_to_draw = 512
    vmin = 0.000000000000001
    vmax = 0.000000001
    fft_size = 512
    overlap = 0.90
    colormap = 'plasma'
    samples_per_bin = fft_size - int(fft_size * overlap)
    fig, (spike_ax, des_ax) = plt.subplots(1, 2, figsize=(8, 3), dpi=300)

    # DES with spike plot
    spike_arch = np.load(spike_path, allow_pickle=True)[spike_index]
    spike_offset = 14
    print(len(spike_arch))
    print(get_stft(spike_arch, show_plot=False).shape[1]*samples_per_bin)
    spike_arch = get_stft(spike_arch, show_plot=False, fft_size=fft_size, overlap_rate=overlap)[:, spike_offset:spike_offset+bins_to_draw]
    spike_len = spike_arch.shape[1]
    spike_arch = np.fft.fftshift(spike_arch, axes=0)
    n = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
    spike_ax.imshow(spike_arch, norm=n, interpolation='bicubic', aspect='auto', cmap=colormap)

    # Fix y axis
    freqs = np.fft.fftshift(np.fft.fftfreq(512, d=1.0/fs))
    spike_ax.set_yticks([0, 128, 256, 384, 512])
    freqs = list(freqs)
    freqs.append(fs/2)
    spike_ax.set_yticklabels([int(freqs[x] / 1e6) for x in spike_ax.get_yticks()])

    # Fix x axis
    spike_ax.set_xlim(0, spike_len)
    spike_ax.set_xticks([x / samples_per_bin for x in [0, 5000, 10000, 15000, 20000, 25000]])
    spike_ax.set_xticklabels([f"{int(x*samples_per_bin):,d}" for x in spike_ax.get_xticks()])
    spike_ax.set_xlabel("Time (samples)")
    spike_ax.set_ylabel("Frequency (MHz)")
    spike_ax.set_title("OpenSSL DES (occluded)")

    # Draw box
    draw_box(spike_ax, 102, 232, "")

    # -----------

    # DES plot
    des_arch = np.load(des_path, allow_pickle=True)[des_index]
    des_arch = get_stft(des_arch, show_plot=False, fft_size=fft_size, overlap_rate=overlap)[:, 0:bins_to_draw]
    des_len = des_arch.shape[1]
    des_arch = np.fft.fftshift(des_arch, axes=0)
    n = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
    des_ax.imshow(des_arch, norm=n, interpolation='bicubic', aspect='auto', cmap=colormap)

    # Fix y axis
    freqs = np.fft.fftshift(np.fft.fftfreq(512, d=1.0/fs))
    des_ax.set_yticks([0, 128, 256, 384, 512])
    freqs = list(freqs)
    freqs.append(fs/2)
    des_ax.set_yticklabels([int(freqs[x] / 1e6) for x in des_ax.get_yticks()])

    # Fix x axis
    des_ax.set_xlim(0, des_len)
    des_ax.set_xticks([x / samples_per_bin for x in [0, 5000, 10000, 15000, 20000, 25000]])
    des_ax.set_xticklabels([f"{int(x*samples_per_bin):,d}" for x in des_ax.get_xticks()])
    des_ax.set_xlabel("Time (samples)")
    des_ax.set_ylabel("Frequency (MHz)")
    des_ax.set_title("OpenSSL DES")

    # Draw box
    draw_box(des_ax, 96, 174, "")

    # Render
    plt.tight_layout()
    with PdfPages('desspikeexample.pdf') as pdf:
        pdf.savefig()


parser = argparse.ArgumentParser(description="Draw DES problem figure.")
args = parser.parse_args()

args.spike_path = "./datasets/nodemcu-random-train2/2020-02-17_11-21-00_296506_traces.npy"
args.des_path = "./datasets/nodemcu-random-train2/2020-02-17_11-21-00_296506_traces.npy"

spike_meta = load_meta(args.spike_path.replace("_traces.npy", "_meta.p"))
print_des_indices(spike_meta)

generate_graph(args.spike_path, args.des_path, spike_index=10, des_index=14)
