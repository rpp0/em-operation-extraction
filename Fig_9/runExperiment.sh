#!/bin/sh

cp ../best_cnn_bb.h5 /tmp/
../autorun.sh "train_nn.py" "testbb best_cnn abs" "cm-seg-dl-abs-best_cnn"
