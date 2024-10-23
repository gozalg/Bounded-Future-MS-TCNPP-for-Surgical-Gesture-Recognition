#!/bin/bash
#--------- USER INPUTS ---------
# DATASET choices: [VTS, JIGSAWS, SAR_RARP50, MultiBypass140]
DATASET=$1
# TASK choices: [steps, phases, gestures]
TASK=$2
BASE_PATH=/data/home/gabrielg/BoundedFuture++/Bounded_Future_from_GIT
TASKS_PATH=${BASE_PATH}/tasks_2D
DATA_PATH=${BASE_PATH}/data
# SPLIT choices: [0, 1, 2, 3, 4] for VTS, MultiBypass140, SAR_RARP50, [0, 1, 2, 3, 4, 5, 6, 7], for JIGSAWS
SPLIT=$3
#-------------------------------------------------
if [ ${DATASET} == "VTS" ]; then
    # FPS=30
    # LABEL_HZ=30
    CLASSES_N=6
    TASK=gestures
    SMP_STEP=6
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_side
    DIR_SUFFIX=${DATASET}/${TASK}
elif [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=10
    TASK=gestures
    SMP_STEP=80
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/${TASK}
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    TASK=gestures
    CLASSES_N=8
    SMP_STEP=6 # 60
    IMG_TMP={:09d}.png
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
elif [ ${DATASET} == "MultiBypass140" ]; then
    FPS=25
    LABEL_HZ=25
    TASK=${TASK}
    if [ ${TASK} == "steps" ]; then
        CLASSES_N=46
    elif [ ${TASK} == "phases" ]; then
        CLASSES_N=14
    else
        echo "Invalid argument (TASK): Choices: [steps, phases]"
        exit
    fi
    SMP_STEP=30
    IMG_TMP={}_{:08d}.jpg
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50, MultiBypass140]"
    exit
fi
# Check if SPLIT is a non-negative integer and belongs to the correct range
if [[ ${SPLIT} -lt 0 ]]; then
    echo "Invalid argument (SPLIT): SPLIT must be a non-negative integer.\n"
    exit
fi
if [[ "${DATASET}" == "JIGSAWS" ]]; then
    if [[ ${SPLIT} -gt 7 ]]; then
        echo "Invalid argument (SPLIT): # SPLIT choices: [0, 1, 2, 3, 4, 5, 6, 7] for JIGSAWS\n"
        exit
    fi
else
    if [[ ${SPLIT} -gt 4 ]]; then
        echo "Invalid argument (SPLIT): # SPLIT choices: [0, 1, 2, 3, 4] for VTS, MultiBypass140, SAR_RARP50\n"
        exit
    fi
fi
#-------------------------------------------------
SMP_PER_CLASS=400
EPOCHS_NUM=100
# SMP_PER_EPOCH=$(( CLASSES_N * SMP_PER_CLASS ))
SRV=so01
script_name=${DATASET}_Features_${task}${SPLIT}
#-------------------------------------------------

# This script is used train and create the features for the dataset for each split:
python ${BASE_PATH}/2D_trainer.py   \
            --wandb true \
            --eval_freq 1 \
            --image_tmpl ${IMG_TMP} \
            --dataset ${DATASET} \
            --task ${TASK} \
            --num_classes ${CLASSES_N} \
            --number_of_samples_per_class ${SMP_PER_CLASS} \
            --val_sampling_step ${SMP_STEP} \
            --epochs ${EPOCHS_NUM} \
            --data_path ${DATA_PATH}/${DATASET}/frames \
            --transcriptions_dir ${DATA_PATH}/${DATASET}/transcriptions \
            --out ${BASE_PATH}/output/feature_extractor \
            --exp ${DATASET} \
            --project_name ${DATASET}_Feature_Extractor_${TASK}_${SRV} \
            --split_num ${SPLIT} \
            --workers 28
            # --video_lists_dir ${BASE_PATH}/data/${DATASET}/Splits \