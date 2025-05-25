#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 64
#SBATCH --mem=128g
#SBATCH --exclude=n312
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriel.gozal@gmail.com
#--------------------- User ----------------------
DATASET=SAR-RARP50
BASE_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT
TASKS_PATH=/rg/laufer_prj/gabrielg/BoundedFuture++/tasks
#-------------------------------------------------
script_name=${DATASET}_unpack
mkdir -p ${TASKS_PATH}/logs
srun    -G 1 -o ${TASKS_PATH}/logs/${script_name}_out_%j.out \
        -e ${TASKS_PATH}/logs/${script_name}_err_%j.err \
        --container-image /rg/laufer_prj/gabrielg/BoundedFuture++/Bounded_Future_from_GIT/nvidia+pytorch+24.04-py3.sqsh \
        --container-mounts /rg/laufer_prj/gabrielg/:/rg/laufer_prj/gabrielg \
        python3 ${BASE_PATH}/data/SAR-RARP50/SAR_RARP50-evaluation/scripts/sarrarp50.py \
                unpack \
                ${BASE_PATH}/data/SAR-RARP50 \
                -r