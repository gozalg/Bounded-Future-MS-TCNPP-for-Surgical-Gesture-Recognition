#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 64
#SBATCH --mem=100g
#SBATCH --exclude=n305,n312
#SBATCH --qos=basic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriel.gozal@gmail.com
#--------------------- User ----------------------
ARCH=X3D # choices: [X3D, EfficientNetV2]
ARCH_SIZE=L # choices: X3D: [XS, S, M, L] EfficientNetV2: [S, M, L]
DATASET=${DATASET}
# TASK choices: [steps, phases, gestures]
TASK=${TASK}
BASE_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT
TASKS_PATH=${BASE_PATH}/tasks_3D
DATA_PATH=${BASE_PATH}/data
# SPLIT choices: [0, 1, 2, 3, 4] for VTS, MultiBypass140, SAR_RARP50, [0, 1, 2, 3, 4, 5, 6, 7], for JIGSAWS
SPLIT=${SPLIT}
# JIGSAWS: SPLIT_LIST=(0 1 2 3 4 5 6 7); for SPLIT in "${SPLIT_LIST[@]}"; do DATASET=JIGSAWS; TASK=gestures; echo "DATASET=${DATASET}, TASK=${TASK}, SPLIT=${SPLIT}"; sbatch --export=DATASET=${DATASET},TASK=${TASK},SPLIT=${SPLIT} ./train_FE.sh; done
# JIGSAWS:          for SPLIT in {0..7}; do DATASET=JIGSAWS; TASK=gestures; echo "DATASET=${DATASET}, TASK=${TASK}, SPLIT=${SPLIT}"; sbatch --export=DATASET=${DATASET},TASK=${TASK},SPLIT=${SPLIT} ./train_FE.sh; done
# VTS:              for SPLIT in {0..4}; do DATASET=VTS; TASK=gestures; echo "DATASET=${DATASET}, TASK=${TASK}, SPLIT=${SPLIT}"; sbatch --export=DATASET=${DATASET},TASK=${TASK},SPLIT=${SPLIT} ./train_FE.sh; done
# MultiBypass140:   for SPLIT in {0..4}; do DATASET=MultiBypass140; TASK=steps; echo "DATASET=${DATASET}, TASK=${TASK}, SPLIT=${SPLIT}"; sbatch --export=DATASET=${DATASET},TASK=${TASK},SPLIT=${SPLIT} ./train_FE.sh; done
# MultiBypass140:   for SPLIT in {0..4}; do DATASET=MultiBypass140; TASK=phases; echo "DATASET=${DATASET}, TASK=${TASK}, SPLIT=${SPLIT}"; sbatch --export=DATASET=${DATASET},TASK=${TASK},SPLIT=${SPLIT} ./train_FE.sh; done
# SAR_RARP50:       for SPLIT in {0..4}; do DATASET=SAR_RARP50; TASK=gestures; echo "DATASET=${DATASET}, TASK=${TASK}, SPLIT=${SPLIT}"; sbatch --export=DATASET=${DATASET},TASK=${TASK},SPLIT=${SPLIT} ./train_FE.sh; done
#-------------------------------------------------
if [ ${ARCH}  == "X3D" ]; then
    train_script=3D_trainer
    batch_size=4
elif [ ${ARCH} == "EfficientNetV2" ]; then
    train_script=3D_trainer
    batch_size=32
else
    echo "Invalid argument (ARCH): Choices: [X3D-L, EfficientNetV2]"
    exit
fi
# chech match between arch and arch_size. for each arch check size given is in the list
if [ ${ARCH} == "X3D" ]; then
    arch_size_list=(XS S M L )
elif [ ${ARCH} == "EfficientNetV2" ]; then
    arch_size_list=(S M L)
else
    echo "Invalid argument (ARCH): Choices: [X3D-L, EfficientNetV2]"
    exit
fi
if [[ ! " ${arch_size_list[@]} " =~ " ${ARCH_SIZE} " ]]; then
    echo "Invalid argument (ARCH_SIZE): for ARCH: ${ARCH} Choices: [${arch_size_list[@]}]"
    exit
fi

# Check if DATASET is a valid argument
if [ ${DATASET} == "VTS" ]; then
    # FPS=30
    # LABEL_HZ=30
    CLASSES_N=6
    TASK=gestures
    SMP_STEP=6
    IMG_TMP=img_{:05d}.jpg
    # VID_SUFFIX=_side
    DIR_SUFFIX=${DATASET}/${TASK}
    VID_LIST_SUFFIX=/${TASK}
elif [ ${DATASET} == "JIGSAWS" ]; then
    FPS=30
    LABEL_HZ=30
    CLASSES_N=10
    TASK=gestures
    SMP_STEP=80
    IMG_TMP=img_{:05d}.jpg
    # VID_SUFFIX=_capture2
    DIR_SUFFIX=${DATASET}/${TASK}
    VID_LIST_SUFFIX=/${TASK}
elif [ ${DATASET} == "SAR_RARP50" ]; then
    FPS=60
    LABEL_HZ=10
    TASK=gestures
    CLASSES_N=8
    SMP_STEP=60
    IMG_TMP={:09d}.png
    # VID_SUFFIX=""
    DIR_SUFFIX=${DATASET}
elif [ ${DATASET} == "MultiBypass140" ]; then
    FPS=25
    LABEL_HZ=25
    TASK=${TASK}
    if [ ${TASK} == "steps" ]; then
        CLASSES_N=46
    elif [ ${TASK} == "phases" ]; then
        CLASSES_N=12
    else
        echo "Invalid argument (TASK): Choices: [steps, phases]"
        exit
    fi
    SMP_STEP=30
    IMG_TMP={}_{:08d}.jpg
    # VID_SUFFIX=""
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
SRV=DGX
script_name=${DATASET}_Features_${task}${SPLIT}
#-------------------------------------------------

mkdir -p ${TASKS_PATH}/logs
mkdir -p ${TASKS_PATH}/logs/FeatureExtractor
srun    -G 1 -o ${TASKS_PATH}/logs/FeatureExtractor/${script_name}_%j.log \
        -e ${TASKS_PATH}/logs/FeatureExtractor/${script_name}_%j.log \
        --container-image ${BASE_PATH}/nvidia+pytorch+24.04-py3.sqsh \
        --container-mounts /rg/laufer_prj/gabrielg/:/rg/laufer_prj/gabrielg \
        python3 ${BASE_PATH}/${train_script}.py   \
                --wandb true \
                --eval_freq 1 \
                --image_tmpl "${IMG_TMP}" \
                --dataset "${DATASET}" \
                --task "${TASK}" \
                --num_classes "${CLASSES_N}" \
                --number_of_samples_per_class "${SMP_PER_CLASS}" \
                --val_sampling_step "${SMP_STEP}" \
                --epochs "${EPOCHS_NUM}" \
                --data_path "${DATA_PATH}"/"${DATASET}"/frames \
                --transcriptions_dir "${DATA_PATH}"/"${DATASET}"/transcriptions \
                --out "${BASE_PATH}"/output/feature_extractor \
                --exp "${DATASET}" \
                --project_name "${DATASET}"_Feature_Extractor_"${TASK}"_"${SRV}" \
                --split_num "${SPLIT}" \
                --batch_size "${batch_size}" \
                --resume_exp "${BASE_PATH}"/output/feature_extractor/"${DATASET}"/X3D-L/"${TASK}"_epochs_100/"${SPLIT}" \
                --workers 64