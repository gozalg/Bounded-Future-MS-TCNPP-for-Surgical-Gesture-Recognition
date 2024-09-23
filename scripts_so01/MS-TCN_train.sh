#!/bin/bash
#--------- USER INPUTS ---------
DATASET=$1
TASK=$2
BASE_PATH=/data/home/gabrielg/Bounded_Future_from_GIT
W_MAX=0
GPUS=1
#------------------------------
FEATURE_EXTRRACTOR=2D-EfficientNetV2-m
SRV=so01
SMP_PER_CLASS=400
EPOCHS_NUM=40
EVAL_FREQ=1

if [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=10
    SMP_STEP=80 # 1
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/Suturing
    VID_LIST_SUFFIX=/Suturing
    TASK=Suturing
    # GPU=0
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    CLASSES_N=8
    SMP_STEP=60 # 6
    IMG_TMP={:09d}.png
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
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
    VID_SUFFIX=None
    DIR_SUFFIX=${DATASET}
    # GPU=0
else
    echo "Invalid argument (DATASET): Choices: [JIAGSAWS, SAR_RARP50, MultiBypass140]"
    echo "Usage: FE_EVAL.sh [DATASET] [TASK]"
    exit
fi

# This script is used traines the MS-TCN model with the features extracted by the Feature Extractor model.
python ${BASE_PATH}/FeatureExtractorTrainer.py \
                    --out ${BASE_PATH}/output/feature_extractor \
                    --project_name ${DATASET}_Feature_Extractor_${TASK}_${SRV} \
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
                    --epochs ${EPOCHS_NUM} \
                    --eval_freq ${EVAL_FREQ} \
                    --arch ${ARCH} \
                    --gpu_id ${GPU} \
                    --workers 16
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
                    --num_layers_PG 10 \
                    --num_layers_R 10 \
                    --num_f_maps 128 \
                    --normalization None \
                    --num_R 3 \
                    --sample_rate $((FPS/LABEL_HZ)) \
                    --loss_tau 16 \
                    --loss_lambda 1 \
                    --dropout_TCN 0.5 \
                    --project ${SRV}_BF-MS-TCN_${DATASET}_wmax_${W_MAX} \
                    --upload True
                    # --RR_not_BF_mode True \
                    # --use_gpu_num ${GPUS} \                    