#!/bin/bash
#--------- USER INPUTS ---------
DATASET=$1
BASE_PATH=/data/home/gabrielg/Bounded_Future_from_GIT
#------------------------------
ARCH=2D-EfficientNetV2-m
SMP_PER_CLASS=400
EPOCHS_NUM=3
EVAL_FREQ=1

if [ ${DATASET} == "JIGSAWS" ]; then
    # FPS=30
    # LABEL_HZ=30
    CLASSES_N=10
    SMP_STEP=80
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/Suturing
    VID_LIST_SUFFIX=/Suturing
    TASK=Suturing
    
elif [ ${DATASET} == "SAR_RARP50" ]; then
    # FPS=60
    # LABEL_HZ=10
    CLASSES_N=8
    SMP_STEP=60
    IMG_TMP={:09d}.png
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    TASK=None
elif [ ${DATASET} == "MultiBypass140" ]; then
    # FPS=60
    # LABEL_HZ=10
    CLASSES_N=46
    SMP_STEP=30
    IMG_TMP={}_{:08d}.jpg
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    TASK=None
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50, MultiBypass140]"
    echo "Usage: FE_EVAL.sh [DATASET]"
    exit
fi

# This script is used create the features for the dataset, by splits models.
python ${BASE_PATH}/FeatureExtractorTrainer.py \
                    --dataset ${DATASET} \
                    --task ${TASK} \
                    --num_classes ${CLASSES_N} \
                    --number_of_samples_per_class ${SMP_PER_CLASS} \
                    --val_sampling_step ${SMP_STEP} \
                    --image_tmpl ${IMG_TMP} \
                    --video_suffix ${VID_SUFFIX} \
                    --data_path ${BASE_PATH}/data/${DIR_SUFFIX}/frames \
                    --transcriptions_dir ${BASE_PATH}/data/${DATASET}/transcriptions \
                    --video_lists_dir ${BASE_PATH}/data/${DATASET}/Splits${VID_LIST_SUFFIX} \
                    --out ${BASE_PATH}/output/feature_extractor \
                    --epochs ${EPOCHS_NUM} \
                    --eval_freq ${EVAL_FREQ} \
                    --arch ${ARCH} \
                    --workers 16