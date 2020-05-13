#!/bin/sh

RESULTS_FOLDER=./results/
REGENERATE_DATA=true
SCRIPT_TO_RUN=$1
SCRIPT_ARGS=$2
RESULT_NAME=$3
CUR_DIR=`pwd`

cd ..
echo "Activating environment"
source env/bin/activate

if [ "$REGENERATE_DATA" = true ]; then
    echo "Regenerating result data files"
    echo "Running script to store in ${RESULTS_FOLDER}"
    python ${SCRIPT_TO_RUN} ${SCRIPT_ARGS}
    echo "Deleting h5 files in /tmp/"
    rm -rf /tmp/*.h5
fi

echo "Copying result"
cp ${RESULTS_FOLDER}${RESULT_NAME}.p $CUR_DIR

echo "Generating figures"
python ./calculate_precision_recall.py $CUR_DIR
python ./draw_confusion_matrices.py $CUR_DIR
