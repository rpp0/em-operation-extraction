#!/usr/bin/sh

# All trained models are saved in /tmp/ to avoid overwriting the pretrained models in this directory
# Note: the "_newaugment" suffix is not automatically added, e.g., best_cnn_bb_newaugment.h5 is saved in /tmp/best_cnn_bb.h5

echo "Uncomment the model(s) that you want to retrain"

# -----------
# 1D variants
# -----------

# best_cnn.h5
#./train_nn.py --datasets-root /home/wisec2020/em-fingerprinting/datasets/ train best_cnn abs --epochs 2500

# best_cnn_bb.h5
#./train_nn.py --datasets-root /home/wisec2020/em-fingerprinting/datasets/ trainbb best_cnn abs --epochs 2500

# best_cnn_bb_newaugment.h5
#./train_nn.py --datasets-root /home/wisec2020/em-fingerprinting/datasets/ trainbb best_cnn abs --epochs 2500 --use-newaugment

# -----------
# 2D variants
# -----------

# best_cnn_2d.h5
#./train_nn_2d.py --datasets-root /home/wisec2020/em-fingerprinting/datasets/ train

# best_cnn_bb_2d
#./train_nn_2d.py --datasets-root /home/wisec2020/em-fingerprinting/datasets/ trainbb --epochs 2500

# best_cnn_bb_2d_newaugment
#./train_nn_2d.py --datasets-root /home/wisec2020/em-fingerprinting/datasets/ trainbb --epochs 2500 --use-newaugment

