#!/bin/sh

cp ../best_cnn_bb_2d.h5 /tmp/
../autorun.sh "train_nn_2d.py" "testbb" "cm-seg-dl2d"
