#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem=50g
#SBATCH --exclude=n305,n312
#SBATCH --qos=normal
#--------------------- Setup ---------------------
# Load user profile settings
source ~/.bashrc
# Set environment variables for CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH:/root/miniconda3/condabin:/usr/local/nvm/versions/node/v16.20.2/bin:/root/.local/bin:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
DATE=$(date '+%Y-%m-%d_%H-%M-%S')

# for W_MAX in {1..20}; do 
#     if (( (W_MAX - 1) % 5 == 0 )); then
#         W_MAX=${W_MAX}
#         DATASET=MultiBypass140
#         TASK=phases
#         USE_DYNAMIC_WMAX=False
#         echo "W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}, USE_DYNAMIC_WMAX=${USE_DYNAMIC_WMAX}";
#         sbatch --export=W_MAX=${W_MAX},DATASET=${DATASET},TASK=${TASK},USE_DYNAMIC_WMAX=${USE_DYNAMIC_WMAX} MS-TCN_train.sh; 
#     fi
# done
# R_N_LIST=(0 1 2 3); LAYERS_N_LIST=(2 3 4 5 6 8 10); W_MAX_LIST=(0 1 2 3 6 7 8 10 12 13 14 15 16 17 20); for R_N in "${R_N_LIST[@]}"; do for LAYERS_N in "${LAYERS_N_LIST[@]}"; do for W_MAX in "${W_MAX_LIST[@]}"; do W_MAX=${W_MAX}; DATASET=MultiBypass140; TASK=steps; echo "R_N=${R_N}, LAYERS_N=${LAYERS_N}, W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}"; export R_N=${R_N}; export LAYERS_N=${LAYERS_N}; export W_MAX=${W_MAX}; export DATASET=${DATASET}; export TASK=${TASK}; sbatch ./BF-MS-TCN_train.sh; done; done; done;
# VTS-gestures:             W_MAX_LIST=(0 1 2 3 6 7 8 10 12 13 14 15 16 17 20); for R_N in {3..3}; do for LAYERS_N in {10..10}; do for W_MAX in "${W_MAX_LIST[@]}"; do W_MAX=${W_MAX}; BACKBONE=EfficientNetV2-M; DATASET=VTS; TASK=gestures; echo "R_N=${R_N}, LAYERS_N=${LAYERS_N}, W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}, BACKBONE=${BACKBONE}"; export R_N=${R_N}; export LAYERS_N=${LAYERS_N}; export W_MAX=${W_MAX}; export DATASET=${DATASET}; export TASK=${TASK}; export BACKBONE=${BACKBONE}; sbatch ./BF-MS-TCN_train.sh; done; done; done;
# JIGSAWS-gestures:         W_MAX_LIST=(0 1 2 3 6 7 8 10 12 13 14 15 16 17 20); for R_N in {3..3}; do for LAYERS_N in {10..10}; do for W_MAX in "${W_MAX_LIST[@]}"; do W_MAX=${W_MAX}; BACKBONE=EfficientNetV2-M; DATASET=JIGSAWS; TASK=gestures; echo "R_N=${R_N}, LAYERS_N=${LAYERS_N}, W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}, BACKBONE=${BACKBONE}"; export R_N=${R_N}; export LAYERS_N=${LAYERS_N}; export W_MAX=${W_MAX}; export DATASET=${DATASET}; export TASK=${TASK}; export BACKBONE=${BACKBONE}; sbatch ./BF-MS-TCN_train.sh; done; done; done;
# MultiBypass140-Steps:     W_MAX_LIST=(0 1 2 3 6 7 8 10 12 13 14 15 16 17 20); for R_N in {3..3}; do for LAYERS_N in {10..10}; do for W_MAX in "${W_MAX_LIST[@]}"; do W_MAX=${W_MAX}; BACKBONE=EfficientNetV2-M; DATASET=MultiBypass140; TASK=steps; echo "R_N=${R_N}, LAYERS_N=${LAYERS_N}, W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}, BACKBONE=${BACKBONE}"; export R_N=${R_N}; export LAYERS_N=${LAYERS_N}; export W_MAX=${W_MAX}; export DATASET=${DATASET}; export TASK=${TASK}; export BACKBONE=${BACKBONE}; sbatch ./BF-MS-TCN_train.sh; done; done; done;
# MultiBypass140-phases:    W_MAX_LIST=(0 1 2 3 6 7 8 10 12 13 14 15 16 17 20); for R_N in {3..3}; do for LAYERS_N in {10..10}; do for W_MAX in "${W_MAX_LIST[@]}"; do W_MAX=${W_MAX}; BACKBONE=EfficientNetV2-M; DATASET=MultiBypass140; TASK=phases; echo "R_N=${R_N}, LAYERS_N=${LAYERS_N}, W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}, BACKBONE=${BACKBONE}"; export R_N=${R_N}; export LAYERS_N=${LAYERS_N}; export W_MAX=${W_MAX}; export DATASET=${DATASET}; export TASK=${TASK}; export BACKBONE=${BACKBONE}; sbatch ./BF-MS-TCN_train.sh; done; done; done;
# SAR_RARP50-gestures:      W_MAX_LIST=(0 1 2 3 6 7 8 10 12 13 14 15 16 17 20); for R_N in {3..3}; do for LAYERS_N in {10..10}; do for W_MAX in "${W_MAX_LIST[@]}"; do W_MAX=${W_MAX}; BACKBONE=EfficientNetV2-M; DATASET=SAR_RARP50; TASK=gestures; echo "R_N=${R_N}, LAYERS_N=${LAYERS_N}, W_MAX=${W_MAX}, DATASET=${DATASET}, TASK=${TASK}, BACKBONE=${BACKBONE}"; export R_N=${R_N}; export LAYERS_N=${LAYERS_N}; export W_MAX=${W_MAX}; export DATASET=${DATASET}; export TASK=${TASK}; export BACKBONE=${BACKBONE}; sbatch ./BF-MS-TCN_train.sh; done; done; done;
#--------------------- User ----------------------
#------------------------------
BACKBONE=${BACKBONE}    # options: [X3D-XS, X3D-S, X3D-M, X3D-L, EfficientNetV2-S, EfficientNetV2-M, EfficientNetV2-L]
if [[ ${BACKBONE} == X3D-* ]]; then
    FTR_DIM=192                 # 192 for X3D-*, 1280 for EfficientNetV2-*
elif [[ ${BACKBONE} == EfficientNetV2-* ]]; then
    FTR_DIM=1280                # 192 for X3D-*, 1280 for EfficientNetV2-*
else
    echo "Invalid argument (BACKBONE): Choices: [X3D-XS, X3D-S, X3D-M, X3D-L, EfficientNetV2-S, EfficientNetV2-M, EfficientNetV2-L]"
    exit
fi
#------------------------------
DATASET=${DATASET}      # options: [VTS, JIGSAWS, SAR_RARP50, MultiBypass140]
TASK=${TASK}            # options: [gestures, phases, steps]
RR_or_BF=BF             # RR for RR-MS-TCN ("offline"), BF for BF-MS-TCN ("online")
W_MAX=${W_MAX}          # [0,1,2,3,6,7,8,10,12,13,14,15,16,17,20]
LAYERS_N=${LAYERS_N}    # [2,3,4,5,6,8,10]
R_N=${R_N}              # [0,1,2,3]
echo "FUTURE_EXTRRACTOR=${BACKBONE}"
echo "DATASET=${DATASET}"
echo "TASK=${TASK}"
echo "RR_or_BF=${RR_or_BF}"
echo "W_MAX=${W_MAX}"
echo "LAYERS_N=${LAYERS_N}"
echo "R_N=${R_N}"
#-------------------------------------------------
EVAL_SCHEME=LOUO
GPUS=1
BASE_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT # _Branch_dyn_wmax 
TASKS_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT/tasks_3D
#-------------------------------------------------
if [ ${DATASET} == "JIGSAWS" -o ${DATASET} == "VTS" ]; then
    # FPS=30
    # LABEL_HZ=30
    # CLASSES_N=10
    TASK=gestures
elif [ ${DATASET} == "SAR_RARP50" ]; then
    # FPS=60
    # LABEL_HZ=10
    # CLASSES_N=8
    TASK=gestures
elif [ ${DATASET} == "MultiBypass140" ]; then
    # FPS=25
    # LABEL_HZ=25
    # CLASSES_N=46 for steps 14/12 for phases
    if [ ${TASK} == "steps" -o ${TASK} == "phases" ]; then
        echo "DATASET=${DATASET}, TASK=${TASK}"
    else
        echo "Invalid argument (TASK): Choices: [gestures, phases, steps]"
        exit
    fi
else
    echo "Invalid DATASET: Choices: [VTS, JIAGSAWS, SAR_RARP50, MultiBypass140]"
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
script_name="${DATE}_${DATASET}_${TASK}_${SCRIPT_SUFFIX}_w_max-${W_MAX}_Layers-${LAYERS_N}_Rnum-${R_N}"
#-------------------------------------------------
mkdir -p ${TASKS_PATH}/logs
srun    --container-image ${BASE_PATH}/nvidia+pytorch+24.04-py3.sqsh \
        --container-mounts /rg/laufer_prj/gabrielg/:/rg/laufer_prj/gabrielg \
        -o ${TASKS_PATH}/logs/BF-MS-TCN/${BACKBONE}/${script_name}_%j.log \
        -e ${TASKS_PATH}/logs/BF-MS-TCN/${BACKBONE}/${script_name}_%j.log \
        python3 ${BASE_PATH}/train_experiment.py \
                --dataset ${DATASET} \
                --eval_scheme ${EVAL_SCHEME} \
                --task ${TASK} \
                --feature_extractor ${BACKBONE} \
                --network MS-TCN2 \
                --split all \
                --features_dim ${FTR_DIM} \
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