#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 64
#SBATCH --mem=128g
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
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
#--------------------- User ----------------------
DATASET=JIGSAWS
EVAL_SCHEME=LOUO
W_MAX=2500
GPUS=1
#-------------------------------------------------
FEATURE_EXTRRACTOR=2D-EfficientNetV2-m
script_name=MS_TCN_with_${DATASET}_${EVAL_SCHEME}_task
mkdir -p /rg/laufer_prj/gabrielg/BoundedFuture++/tasks/logs
srun    -G ${GPUS} -o /rg/laufer_prj/gabrielg/BoundedFuture++/tasks/logs/${script_name}_out_%j.out \
        -e /rg/laufer_prj/gabrielg/BoundedFuture++/tasks/logs/${script_name}_err_%j.err \
        --container-image /rg/laufer_prj/gabrielg/prj/rev04a_nvidia+pytorch+24.04-py3.sqsh \
        --container-mounts /rg/laufer_prj/gabrielg/:/workspace \
        python3 /workspace/BoundedFuture++/Bounded_Future_from_GIT/train_experiment.py \
                --dataset ${DATASET} \
                --eval_scheme ${EVAL_SCHEME} \
                --feature_extractor ${FEATURE_EXTRRACTOR} \
                --network MS-TCN2 \
                --split all \
                --features_dim 1280 \
                --lr 0.0010351748096577 \
                --num_epochs 40 \
                --eval_rate 1 \
                --RR_not_BF_mode True \
                --w_max ${W_MAX} \
                --num_layers_PG 10 \
                --num_layers_R 10 \
                --num_f_maps 128 \
                --normalization None \
                --num_R 3 \
                --sample_rate 1 \
                --loss_tau 16 \
                --loss_lambda 1 \
                --dropout_TCN 0.5 \
                --project RR-MS-TCN_${DATASET}_${EVAL_SCHEME}_wmax_${W_MAX}_DGX \
                --upload True
                # --use_gpu_num ${GPUS} \
echo "Running ${script_name}..."