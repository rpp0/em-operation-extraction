# Practical Operation Extraction from Electromagnetic Leakage for Side-Channel Analysis and Reverse Engineering

This repository contains the code for our work presented at WiSec 2020, entitled "Practical Operation Extraction from Electromagnetic Leakage for Side-Channel Analysis and Reverse Engineering". In order to replicate our obtained results, please download the accompanying VM via http://wisecdata.ccs.neu.edu/. In addition to this code, the VM contains all pre-trained models and datasets as well.

## 1. Replicating our results using pre-trained models

1. Change the current working directory to `/home/wisec2020/em-fingerprinting/`.
2. Execute `source env/bin/activate`.
3. Execute the command corresponding to the desired figure in order to generate a `.p` file in `results/`, using the following table as a reference. *Important note*: the ML-based experiments (using either `train_nn.py` or `train_nn_2d.py`) may overwrite the result data files from previous ML-based experiments. It is therefore advised to proceed to step 3 for each experiment individually, rather than performing all commands listed here at once.

| Name                       | `paper_results` file name            | Command          |  Note |
|----------------------------|:---------------------------:|------------------:|-----:|
| Figure 3 confusion matrix  | `cm-roi-norm-abs-norm_corr` | `./fingerprint.py experiment_roi_norm abs norm_corr` | 36.74% acc, 47.23% prec, 37.38% recall |
| Figure 4 confusion matrix  | `cm-roi-norm-abs-sq_diff`   | `./fingerprint.py experiment_roi_norm abs sq_diff` | 10.4% acc, 1.19% prec, 11.06% recall |
| Figure 6 confusion matrix  | `cm-roi-dl-abs-best_cnn`    | `cp best_cnn.h5 /tmp/ && ./train_nn.py test best_cnn abs` | 94.43% acc, 94.93% prec, 94.06% recall |
| Figure 8 confusion matrix  | `cm-roi-dl2d`               | `cp best_cnn_2d.h5 /tmp/ && ./train_nn_2d.py test` | 94.38% acc, 95.92% prec, 94.01% recall |
| Figure 9 confusion matrix  | `cm-seg-dl-abs-best_cnn`    | `cp best_cnn_bb.h5 /tmp/ && ./train_nn.py testbb best_cnn abs` | 74.90% acc, 77.84% prec, 77.87% recall |
| 1D CNN bounding box error  | `dc-seg-dl-abs-best_cnn`    | *Generated together with Figure 9 command.* | 429.81 μs |
| Figure 10 confusion matrix | `cm-seg-dl2d`               | `cp best_cnn_bb_2d.h5 /tmp/ && ./train_nn_2d.py testbb` | 96.47% acc, 96.69% prec, 96.78% recall |
| 2D CNN bounding box error  | `dc-seg-dl2d`    | *Generated together with Figure 10 command.* | 34 μs |
| 4.6.2 1D CNN on Wi-Fi connect snippet | `cm-seg-dl-abs-best_cnn-aug` | `cp best_cnn_bb_newaugment.h5 /tmp/best_cnn_bb.h5 && ./train_nn.py testbbaugment best_cnn abs --use-newaugment` | 38.04% acc, 63.84% prec, 40.44% recall. Since the augmentation performed here is random, results will vary slightly for each test. |
| 4.6.2 2D CNN on Wi-Fi connect snippet | `cm-seg-dl2d-aug` | `cp best_cnn_bb_2d_newaugment.h5 /tmp/best_cnn_bb_2d.h5 && ./train_nn_2d.py testbbaugment --use-newaugment`|  55.29% acc, 78.67% prec, 54.36% recall. Since the augmentation performed here is random, results will vary slightly for each test. |
| 4.6.2 1D CNN (augmented) on random label test dataset | `cm-seg-dl-abs-best_cnn-newaugment`    | `cp best_cnn_bb_newaugment.h5 /tmp/best_cnn_bb.h5 && ./train_nn.py testbb best_cnn abs` | Should yield 81.96% accuracy at step 5. Confusion matrix not shown in paper. |
| 4.6.2 2D CNN (augmented) on random label test dataset | `cm-seg-dl2d-newaugment`               | `cp best_cnn_bb_2d_newaugment.h5 /tmp/best_cnn_bb_2d.h5 && ./train_nn_2d.py testbb` | Should yield 96.86% accuracy at step 5. Confusion matrix not shown in paper. |

4. Verify the results. To do this, please read Section 2 in this document on verification, and then continue. Optionally, generate the `pdf` of the confusion matrices using the generated `.p` file by executing `python draw_confusion_matrices.py results`.

5. Print the accuracy, precision and recall by executing `python calculate_precision_recall.py results`. The values reported in the paper are listed under *macro-averaged precision*, *macro-averaged recall* and *accuracy*.

6. Delete (or manually rename) all results in `results/` to avoid confusion when performing the next experiment. Delete `/tmp/<model name copied via command in step 3>.h5` to free up memory if needed. Repeat step 3.

## 2. Verifying the results: an example

Each experiment saves a `<experiment_name>.p` file in the `results/` folder after completion. This 'results data' file is subsequently used to generate graphs from, to calculate the accuracy from, etc. The original results we obtained and reported in our paper were placed in the `paper_results/` folder as a backup. Since the new results that the reviewer generates are stored in `results/`, they can easily be verified cryptographically.

Example: suppose we want to verify the 96.86% accuracy claim made in Section 4.6.2, for the 2D CNN (augmented) model tested on the 'random label test dataset'. We proceed as follows:

1. Follow steps 1 and 2 detailed in the previous section if not done already, then execute `cp best_cnn_bb_2d_newaugment.h5 /tmp/best_cnn_bb_2d.h5 && ./train_nn_2d.py testbb`.

2. After completion, a result data file called `cm-seg-dl2d.p` will be stored in `results/`.

3. The corresponding `paper_results` filename is `cm-seg-dl2d-newaugment.p` (see table above), so we can do: `sha256sum ./results/cm-seg-dl2d.p ./paper_results/cm-seg-dl2d-newaugment.p`. Expected output:

```
53894464e75724d1411fa6d0f0d5e943bfc5fe52f6e5a7bdd96670bb66a05c11  ./results/cm-seg-dl2d.p
53894464e75724d1411fa6d0f0d5e943bfc5fe52f6e5a7bdd96670bb66a05c11  ./paper_results/cm-seg-dl2d-newaugment.p
```

In case the results are inherently non-deterministic (e.g. for the experiments in 4.6.2, where operations are inserted randomly into noise snippets), the hashes will not match but the results should be close. Finally, the reviewer can optionally proceed to steps 4 through 5, to verify that the final, formatted results and graphs were correctly extracted from the result data files.

## 3. Verifying the dataset counts

In the appendix, the number of examples per class are given. These counts can be verified using the command `./count_classes.py <dataset_name>`. For example:

```
./count_classes.py nodemcu-random-label-train
```

## 4. Realtime experiment

If you wish to perform realtime classification of EM traces using an SDR and our models, the following steps can be followed:

1. If not done already, install [EMMA](https://github.com/), which is a framework for capturing EM traces using SDR. EMMA is already installed on the virtual machine.
2. Copy `best_cnn_bb_2d.h5` or `best_cnn_bb.h5` to `/tmp/`.
3. Connect a NodeMCU, and make sure it can be accessed on the VM at `/dev/ttyUSB0`. The custom firmware should be uploaded to the NodeMCU, so that it can communicate with EMMA.
4. Run `./realtime_capture.py`.

## 5. Extra results

Due to space constraints, we limited the inclusion of confusion matrices in the paper to the most interesting ones. Some extra results can be found in the `paper_results` folder.

## 6. Performing new measurements

## 7. Retraining models
