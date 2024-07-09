#!/bin/bash
#--------- USER INPUTS ---------
DATASET=$1
#------------------------------
ARCH=2D-EfficientNetV2-m

if [ ${DATASET} == "JIGSAWS" ]; then
    # FPS=30
    # LABEL_HZ=30
    # NUM_CLASS=10
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/Suturing
    VID_LIST_SUFFIX=/Suturing
    TASK=Suturing
elif [ ${DATASET} == "SAR_RARP50" ]; then
    # FPS=60
    # LABEL_HZ=10
    # NUM_CLASS=8
    IMG_TMP={:09d}.png
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    TASK=None
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50]"
    echo "Usage: FE_EVAL.sh [DATASET]"
    exit
fi

# This script is used create the features for the dataset, by splits models.
python FeaturesCreate.py    --dataset ${DATASET} \
                            --task ${TASK} \
                            --image_tmpl ${IMG_TMP} \
                            --video_suffix ${VID_SUFFIX} \
                            --data_path /data/home/gabrielg/Bounded_Future_from_GIT/data/${DIR_SUFFIX}/frames \
                            --video_lists_dir /data/home/gabrielg/Bounded_Future_from_GIT/data/${DATASET}/Splits${VID_LIST_SUFFIX} \
                            --pretrain_path /data/home/gabrielg/Bounded_Future_from_GIT/output/feature_extractor/${DATASET}/${ARCH} \
                            --out /data/home/gabrielg/Bounded_Future_from_GIT/output/features/${DATASET}/${ARCH} \
                            --arch ${ARCH} \
                            --workers 16