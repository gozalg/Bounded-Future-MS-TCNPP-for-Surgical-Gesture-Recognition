#!/bin/bash
#--------- USER INPUTS ---------
BASE_PATH=/data/home/gabrielg/BoundedFuture++/Bounded_Future_from_GIT
DATASET=$1              # options: [VTS, JIGSAWS, SAR_RARP50, MultiBypass140]
TASK=$2                 # options: [gestures, phases, steps]
GPUS=1
RR_or_BF=BF             # RR for RR-MS-TCN ("offline"), BF for BF-MS-TCN ("online")
W_MAX=20                # [0,1,2,3,6,7,8,10,12,13,14,15,16,17,20]
LAYERS_N=10             # [2,3,4,5,6,8,10]
R_N=3                   # [0,1,2,3]
#------------------------------
FEATURE_EXTRRACTOR=2D-EfficientNetV2-m
SRV=so01
EPOCHS_NUM=40
EVAL_FREQ=1
if [ ${DATASET} == "VTS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=6
    SMP_STEP=6 # 6
    IMG_TMP=img_{:05d}.jpg
    TASK=gestures
    # GPU=0
elif [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=10
    SMP_STEP=80 # 1
    IMG_TMP=img_{:05d}.jpg
    TASK=gestures
    # GPU=0
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    CLASSES_N=8
    SMP_STEP=60 # 6
    IMG_TMP={:09d}.png
    TASK=gesture
    # GPU=0
elif [ ${DATASET} == "MultiBypass140" ]; then
    FPS=25
    LABEL_HZ=25
    if [ ${TASK} == "steps" ]; then
        CLASSES_N=46
    elif [ ${TASK} == "phases" ]; then
        CLASSES_N=14
    else
        echo "Invalid argument (TASK): Choices: [Suturing, gesture, steps, phases]"
        echo "Usage: FE_EVAL.sh [DATASET] [TASK]"
        exit
    fi
    SMP_STEP=30 # 1
    IMG_TMP={}_{:08d}.jpg
    # GPU=0
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50, MultiBypass140]"
    echo "Usage: FE_EVAL.sh [DATASET] [TASK]"
    exit
fi

# This script is used traines the MS-TCN model with the features extracted by the Feature Extractor model.
python ${BASE_PATH}/train_experiment.py \
                    --dataset ${DATASET} \
                    --feature_extractor ${FEATURE_EXTRRACTOR} \
                    --network MS-TCN2 \
                    --split all \
                    --features_dim 1280 \
                    --lr 0.0010351748096577 \
                    --num_epochs ${EPOCHS_NUM} \
                    --eval_rate ${EVAL_FREQ} \
                    --w_max ${W_MAX} \
                    --num_layers_PG ${LAYERS_N} \
                    --num_layers_R ${LAYERS_N} \
                    --num_f_maps 128 \
                    --normalization None \
                    --num_R ${R_N} \
                    --sample_rate $((FPS/LABEL_HZ)) \
                    --loss_tau 16 \
                    --loss_lambda 1 \
                    --dropout_TCN 0.5 \
                    --project ${SRV}_BF-MS-TCN_${DATASET}_wmax_${W_MAX} \
                    --upload True
                    # --RR_not_BF_mode True \
                    # --use_gpu_num ${GPUS} \                    