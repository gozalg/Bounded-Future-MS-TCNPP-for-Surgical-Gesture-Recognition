#!/bin/bash
#--------- USER INPUTS ---------
DATASET=$1
BASE_PATH=/data/home/gabrielg/Bounded_Future_from_GIT
#------------------------------

if [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=10
    # SMP_STEP=80
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/Suturing
    VID_LIST_SUFFIX=/Suturing
    TASK=Suturing
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    CLASSES_N=8
    # SMP_STEP=60
    IMG_TMP={:09d}.png
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    TASK=None
elif [ ${DATASET} == "MultiBypass140" ]; then
    FPS=25
    LABEL_HZ=25
    CLASSES_N=46
    # SMP_STEP=30
    IMG_TMP={}_{:08d}.jpg
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    TASK=None
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50, MultiBypass140]"
    echo "Usage: FE_EVAL.sh [DATASET]"
    exit
fi

# This script is used to evaluate the performance of the model on the test set.
python ${BASE_PATH}/FeatureExtractorEval.py  \
                        --dataset ${DATASET} \
                        --num_classes ${CLASSES_N} \
                        --val_sampling_step $((FPS/LABEL_HZ)) \
                        --image_tmpl ${IMG_TMP} \
                        --video_lists_dir ${BASE_PATH}/data/${DATASET}/Splits${VID_LIST_SUFFIX} \
                        --data_path ${BASE_PATH}/data/${DIR_SUFFIX}/frames \
                        --transcriptions_dir ${BASE_PATH}/data/${DIR_SUFFIX}/transcriptions \
                        --video_suffix ${VID_SUFFIX} \
                        --workers 16 \
                        --task ${TASK}

