import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.text as text
import argparse
import sys
from matplotlib.backends.backend_pdf import PdfPages
from emma.processing.dsp import butter_filter

class SampleLineBuilder():
    def __init__(self, fs, axis, height=0):
        self.line_styles = ['-', ':']
        self.line_style_index = 0
        self.fs = fs
        self.axis = axis
        self.fontprop = matplotlib.font_manager.FontProperties(family="Linux Libertine", size=10)
        self.height = height

    def _next_style(self):
        style = self.line_styles[self.line_style_index]
        self.line_style_index = (self.line_style_index + 1) % len(self.line_styles)
        return style

    def next_line(self, offset, length, annotation=""):
        start_sample = offset
        end_sample = offset+length

        # Generate line
        line = lines.Line2D([start_sample/self.fs, end_sample/self.fs], [self.height, self.height], lw=1, color='black', axes=self.axis, linestyle=self._next_style())

        # Generate text
        ptext = text.Text((start_sample + (length / 2.0))/self.fs, self.height, annotation, ha='center', va='bottom', axes=self.axis, fontproperties=self.fontprop)

        # Add them
        self.axis.add_line(line)
        self.axis.add_artist(ptext)

        # Return offset for next segment
        return end_sample


def delete_border(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)

def normalize(a):
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()

def generate_graph(am_path, stft_path):
    # Setup
    plt.rcParams.update({
        'font.size': 10,
        'font.sans-serif': ["Linux Libertine"],
    })
    fig, (am_ax, stft_ax) = plt.subplots(2, 1, figsize=(8, 3), dpi=300)

    # AM plot
    am_arch = np.load(am_path)
    am_arch = butter_filter(am_arch, cutoff=0.05)[0:-10]
    am_arch = normalize(am_arch)

    line_builder = SampleLineBuilder(1, am_ax, height=-0.2)
    offset = 0
    arch_len = len(am_arch)
    offset = line_builder.next_line(offset, arch_len // 4, "HMAC-SHA1")
    offset = line_builder.next_line(offset, arch_len // 4, "HMAC-SHA1")
    offset = line_builder.next_line(offset, arch_len // 4, "HMAC-SHA1")
    offset = line_builder.next_line(offset, arch_len // 4, "HMAC-SHA1")
    am_ax.set_ylabel("Amplitude (unitless)")


    am_ax.plot(am_arch, linewidth=0.7, color="black")
    am_ax.set_ylim([-0.21, 1.1])
    #am_ax.set_xlim(-500, arch_len + 500)
    am_ax.set_xlim(0, arch_len)
    #am_ax.get_xaxis().set_visible(False)
    am_ax.set_xticklabels(['' for x in am_ax.get_xticks()])
    am_ax.spines['bottom'].set_visible(False)

    # STFT plot
    fs = 56.0e6
    stft_arch = np.load(stft_path)
    stft_len = stft_arch.shape[1]
    stft_arch = np.fft.fftshift(stft_arch, axes=0)
    n = matplotlib.colors.LogNorm(vmin=0.000000000000001, vmax=0.0000000001, clip=False)
    stft_ax.imshow(stft_arch, norm=n, interpolation='bicubic', aspect='auto', cmap='plasma')

    # Fix y axis
    freqs = np.fft.fftshift(np.fft.fftfreq(512, d=1.0/fs))
    stft_ax.set_yticks([0, 128, 256, 384, 512])
    freqs = list(freqs)
    freqs.append(fs/2)
    stft_ax.set_yticklabels([int(freqs[x] / 1e6) for x in stft_ax.get_yticks()])

    # Fix x axis
    stft_ax.set_xticks([int((x/arch_len)*stft_len) for x in am_ax.get_xticks()])
    stft_ax.set_xticklabels([f"{int(x):,d}" for x in am_ax.get_xticks()])
    stft_ax.set_xlim(0, stft_len)
    stft_ax.set_xlabel("Time (samples)")
    stft_ax.set_ylabel("Frequency (MHz)")

    # Render
    plt.tight_layout()
    with PdfPages('sha1prfexample.pdf') as pdf:
        pdf.savefig()



parser = argparse.ArgumentParser(description="Draw arch figure.")
parser.add_argument("am_path", type=str, help="Path to AM signal.")
parser.add_argument("stft_path", type=str, help="Path to STFT signal.")
args = parser.parse_args()

generate_graph(args.am_path, args.stft_path)
