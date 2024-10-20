#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 64
#SBATCH --mem=200g
#SBATCH --exclude=n305,n312
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriel.gozal@gmail.com
#--------------------- Setup ---------------------
# Load user profile settings
source ~/.bashrc
# Set environment variables for CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH:/root/miniconda3/condabin:/usr/local/nvm/versions/node/v16.20.2/bin:/root/.local/bin:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#--------- USER INPUTS ---------
# BASE_PATH=/data/home/gabrielg/BoundedFuture++/Bounded_Future_from_GIT # for [srv != "DGX"]
BASE_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT # for [srv == "DGX"]
TASKS_PATH=${BASE_PATH}/tasks_2D
SRV=DGX                 # options: [DGX, so01, so-srv]
DATASET=VTS             # options: [VTS, JIGSAWS, SAR_RARP50, MultiBypass140]   $1 for [srv != "DGX"]
TASK=gestures           # options: [gestures, phases, steps]                    $2 for [srv != "DGX]
GPUS=1
RR_or_BF=BF             # RR for RR-MS-TCN ("offline"), BF for BF-MS-TCN ("online")
W_MAX=20                # [0,1,2,3,6,7,8,10,12,13,14,15,16,17,20]
LAYERS_N=10             # [2,3,4,5,6,8,10]
R_N=3                   # [0,1,2,3]
#------------------------------
FEATURE_EXTRRACTOR=2D-EfficientNetV2-m
EPOCHS_NUM=40
EVAL_FREQ=1
if [ ${DATASET} == "VTS" ]; then
    # FPS=30
    # LABEL_HZ=30
    # CLASSES_N=6
    # SMP_STEP=6 # 6
    # IMG_TMP=img_{:05d}.jpg
    TASK=gestures
    # GPU=0
elif [ ${DATASET} == "JIGSAWS" ]; then
    # FPS=30
    # LABEL_HZ=30
    # CLASSES_N=10
    # SMP_STEP=80 # 1
    # IMG_TMP=img_{:05d}.jpg
    TASK=gestures
    # GPU=0
elif [ ${DATASET} == "SAR_RARP50" ]; then
    # FPS=60
    # LABEL_HZ=10
    # CLASSES_N=8
    # SMP_STEP=60 # 6
    # IMG_TMP={:09d}.png
    TASK=gestures
    # GPU=0
elif [ ${DATASET} == "MultiBypass140" ]; then
    # FPS=25
    # LABEL_HZ=25
    # if [ ${TASK} == "steps" ]; then
    #     CLASSES_N=46
    # elif [ ${TASK} == "phases" ]; then
    #     CLASSES_N=14
    # else
    if [ ${TASK} == "steps" -o ${TASK} == "phases" ]; then
        continue
    else
        echo "Invalid argument (TASK): Choices: [gestures, steps, phases]"
        echo "Usage: FE_EVAL.sh [DATASET] [TASK]"
        exit
    fi
    # SMP_STEP=30 # 1
    # IMG_TMP={}_{:08d}.jpg
    # GPU=0
else
    echo "Invalid argument (DATASET): Choices: [VTS, JIAGSAWS, SAR_RARP50, MultiBypass140]"
    echo "Usage: FE_EVAL.sh [DATASET] [TASK]"
    exit
fi
#-------------------------------------------------
if [ ${RR_or_BF} == "RR" ]; then
    SCRIPT_SUFFIX=RR
elif [ ${RR_or_BF} == "BF" ]; then
    SCRIPT_SUFFIX=BF
else
    echo "Invalid RR_or_BF: Choices: [RR, BF]"
    exit
fi
script_name=${DATASET}_${TASK}_${SCRIPT_SUFFIX}_w_max-${W_MAX}_Layers-${LAYERS_N}_Rnum-${R_N}
#-------------------------------------------------
# This script is used traines the MS-TCN model with the features extracted by the Feature Extractor model.
if [ ${SRV} == "DGX" ]; then
    mkdir -p ${TASKS_PATH}/logs
    srun    --container-image ${BASE_PATH}/nvidia+pytorch+24.04-py3.sqsh \
            --container-mounts /rg/laufer_prj/gabrielg/:/rg/laufer_prj/gabrielg \
            -o ${TASKS_PATH}/logs/MS-TCN/${script_name}_%j.log \
            -e ${TASKS_PATH}/logs/MS-TCN/${script_name}_%j.log \
            python3 ${BASE_PATH}/train_experiment.py \
                    --dataset ${DATASET} \
                    --eval_scheme ${EVAL_SCHEME} \
                    --task ${TASK} \
                    --feature_extractor ${FEATURE_EXTRRACTOR} \
                    --network MS-TCN2 \
                    --split all \
                    --features_dim 1280 \
                    --lr 0.0010351748096577 \
                    --num_epochs 40 \
                    --eval_rate 1 \
                    --w_max ${W_MAX} \
                    --num_layers_PG ${LAYERS_N} \
                    --num_layers_R ${LAYERS_N} \
                    --num_f_maps 128 \
                    --normalization None \
                    --num_R ${R_N} \
                    --sample_rate 1 \
                    --RR_or_BF_mode ${RR_or_BF} \
                    --loss_tau 16 \
                    --loss_lambda 1 \
                    --dropout_TCN 0.5 \
                    --project ${script_name} \
                    --upload True
                    # --use_gpu_num ${GPUS} \
    echo "Running ${script_name}..."
else
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
                        --sample_rate 1 \
                        --loss_tau 16 \
                        --loss_lambda 1 \
                        --dropout_TCN 0.5 \
                        --project ${script_name} \
                        --upload True
                        # --project ${SRV}_BF-MS-TCN_${DATASET}_wmax_${W_MAX} \
                        # --sample_rate $((FPS/LABEL_HZ)) \
                        # --RR_not_BF_mode True \
                        # --use_gpu_num ${GPUS} \  
fi
