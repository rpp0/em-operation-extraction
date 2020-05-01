
import pickle
import numpy
import sys
import os
from common import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Based on https://github.com/DTrimarchi10/confusion_matrix
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None, saveAs=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''
    # Config stuff
    prev_font = plt.rcParams["font.sans-serif"]
    plt.rcParams['font.sans-serif'] = ["Linux Libertine"]
    plt.rcParams['font.size'] = 16

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    # Fix axis bug
    plt.gca().set_ylim([0, len(categories)])
    plt.gca().invert_yaxis()

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    if (saveAs != None):
        plt.savefig(saveAs, bbox_inches='tight')

    plt.cla()
    plt.rcParams["font.sans-serif"] = prev_font


if (len(sys.argv) < 2):
    print("python draw_confusion_matrices.py <path_to_folder>")
    exit(1)

pPath = sys.argv[1]

for file in os.listdir(pPath):
    if file.endswith(".p"):
        pFile = os.path.join(pPath, file)
        print("Found " + file + ". Plotting ...")
        a = open(pFile, "rb")
        data = pickle.load(a)

        if type(data) != ConfusionMatrix:
            continue

        # Exceptions: do not plot "noise" class for a few matrices
        if file == "cm-roi-norm-abs-norm_corr.p" or file == "cm-roi-norm-abs-sq_diff.p":
            print("Removing last row / col for %s" % file)
            data.matrix = data.matrix[:-1, :-1]  # Remove "noise" class (last row/col)

        a.close()
        categories = []
        xLen = data.matrix.shape[0]
        yLen = data.matrix.shape[1]
        for i in range(0, xLen):
            categories.append(int_to_op[i])

        make_confusion_matrix(data.matrix, saveAs=pFile+"df", categories=categories, percent=False, sum_stats=False, cmap="binary")
        print("Done plotting.")

print("Done!")