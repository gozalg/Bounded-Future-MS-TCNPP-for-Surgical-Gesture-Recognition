#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 64
#SBATCH --mem=200g
#SBATCH --exclude=n305,n312
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriel.gozal@gmail.com
#--------------------- User ----------------------
DATASET=MultiBypass140
# TASK choices: [steps, phases, gestures, Sutruring]
TASK=phases
BASE_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT
TASKS_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/tasks
SPLIT=0
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
    VID_LIST_SUFFIX=/${TASK}
elif [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=10
    TASK=gestures
    SMP_STEP=80
    IMG_TMP=img_{:05d}.jpg
    VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/${TASK}
    VID_LIST_SUFFIX=/${TASK}
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    TASK=gestures
    CLASSES_N=8
    SMP_STEP=60
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
#-------------------------------------------------
SMP_PER_CLASS=400
EPOCHS_NUM=100
# SMP_PER_EPOCH=$(( CLASSES_N * SMP_PER_CLASS ))
SRV=DGX
script_name=${DATASET}_Features_${task}${SPLIT}
#-------------------------------------------------

mkdir -p ${TASKS_PATH}/logs
mkdir -p ${TASKS_PATH}/logs/FeatureExtractor
srun    -G 1 -o ${TASKS_PATH}/logs/FeatureExtractor/${script_name}_%j.log \
        -e ${TASKS_PATH}/logs/FeatureExtractor/${script_name}_%j.log \
        --container-image ${BASE_PATH}/nvidia+pytorch+24.04-py3.sqsh \
        --container-mounts /rg/laufer_prj/gabrielg/:/rg/laufer_prj/gabrielg \
        python3 ${BASE_PATH}/2D_trainer.py   \
                --wandb true \
                --eval_freq 1 \
                --image_tmpl ${IMG_TMP} \
                --video_suffix ${VID_SUFFIX} \
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
                --workers 64
                # --video_lists_dir ${BASE_PATH}/data/${DATASET}/Splits \