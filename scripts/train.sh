#!/bin/bash
MODEL=txt_scan_res18
LR=1e-5
BATCH=16
EPOCH=100
ATTN=False
JOB=${MODEL}_lr_${LR}_batch_${BATCH}
SAVE_FOLDER=./checkpoints/${JOB}
python ./exp/train.py train \
     --lr=${LR} \
     --save_folder=${SAVE_FOLDER} \
     --batch_size=${BATCH} \
     --epoches=${EPOCH} \
     --model=${MODEL} \
     --attn=${ATTN}