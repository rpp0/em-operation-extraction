
import pickle
import numpy as np
import sys
import common
import os
from common import ConfusionMatrix


if (len(sys.argv) < 2):
    print("python calc.py <path_to_folder>")
    exit(1)

pPath = sys.argv[1]
for file in os.listdir(pPath):
    if file.endswith(".p"):
        pFile = os.path.join(pPath, file)
        print(file + ":")
        a = open(pFile,"rb")
        data = pickle.load(a)

        if type(data) != ConfusionMatrix:
            continue

        a.close()
        yLen = data.matrix.shape[0]
        xLen = data.matrix.shape[1]
        assert(yLen == xLen)
        print(data.matrix)
        allTP = 0
        allFP = 0
        allFN = 0
        allPrec = 0
        allRecall = 0
        numberClasses = yLen
        round_nums = 5
        for y in range(0,yLen):
            TP = 0
            FP = 0
            FN = 0
            for x in range(0,xLen):
                if (x == y):
                    TP += data.matrix[y][x]
                else:
                    FN += data.matrix[y][x]
            for q in range(0,yLen):
                if (q != y):
                    FP += data.matrix[q][y]

            if (TP+FP != 0):
                prec = round(TP / (TP+FP), round_nums)
            else:
                prec = 0
            if (TP+FN != 0):
                recall = round(TP / (TP + FN), round_nums)
            else:
                recall = 0
            allTP += TP
            allFP += FP
            allFN += FN
            allPrec += prec
            allRecall += recall
            print("Class " + str(y) + ": precision=" + str(prec) + ", recall=" + str(recall))

        microPrec = round(allTP / (allTP+allFP),round_nums)
        microRecall = round(allTP / (allTP + allFN),round_nums)
        macroPrec = round (allPrec / numberClasses,round_nums)
        macroRecall = round (allRecall / numberClasses,round_nums)
        f1Score = round(2 * ((allPrec/numberClasses * allRecall/numberClasses) / (allPrec/numberClasses + allRecall/numberClasses)), round_nums)
        print("Micro-average-precision=" + str(microPrec) + ",  micro-average-recall=" + str(microRecall))
        print("Macro-average-precision=" + str(macroPrec) + ",  macro-average-recall=" + str(macroRecall))
        print("F1-score=" + str(f1Score))
        print("Accuracy=%.5f" % (np.sum(np.diagonal(data.matrix)) / np.sum(data.matrix)))

        dc_file = pFile.replace("cm", "dc")
        if os.path.exists(dc_file):
            with open(dc_file,"rb") as f:
                data = pickle.load(f)
                data.print()
