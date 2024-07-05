#!/bin/bash
#--------- USER INPUTS ---------
DATASET=$1
#------------------------------

if [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    NUM_CLASS=10
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/Suturing
    VID_LIST_SUFFIX=/Suturing
    TASK=Suturing
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    NUM_CLASS=8
    IMG_TMP={:09d}.png
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    TASK=None
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50]"
    echo "Usage: FE_EVAL.sh [DATASET]"
    exit
fi

# This script is used to evaluate the performance of the model on the test set.
python FeatureExtractorEval.py  --dataset ${DATASET} \
                                --num_classes ${NUM_CLASS} \
                                --val_sampling_step $((FPS/LABEL_HZ)) \
                                --image_tmpl ${IMG_TMP} \
                                --video_lists_dir /data/home/gabrielg/Bounded_Future_from_GIT/data/${DATASET}/Splits${VID_LIST_SUFFIX} \
                                --data_path /data/home/gabrielg/Bounded_Future_from_GIT/data/${DIR_SUFFIX}/frames \
                                --transcriptions_dir /data/home/gabrielg/Bounded_Future_from_GIT/data/${DIR_SUFFIX}/transcriptions \
                                --video_suffix ${VID_SUFFIX} \
                                --workers 16 \
                                --task ${TASK}

