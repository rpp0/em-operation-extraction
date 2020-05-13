#!/bin/sh

cp ../best_cnn_2d.h5 /tmp/
../autorun.sh "train_nn_2d.py" "test" "cm-roi-dl2d"
