#!/usr/bin/env python
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='Plot val loss of model over time')
parser.add_argument('path', type=str, help='Path to p file')
args, unknown = parser.parse_known_args()

with open("%s" % args.path, "rb") as f:
    data = pickle.load(f)

plt.plot(data)
plt.show()

