#!/bin/sh

cp ../best_cnn.h5 /tmp/
../autorun.sh "train_nn.py" "test best_cnn abs" "cm-roi-dl-abs-best_cnn"
