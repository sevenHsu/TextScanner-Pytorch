#!/bin/bash
BATCH=1
ATTN=False
MODEL=txt_scan_res18
LOAD_PATH=checkpoints/txt_scan_res18_lr_1e-5_batch_16/best_val_error.pth
python ./exp/test.py test \
     --load_model_path=${LOAD_PATH} \
     --batch_size=${BATCH} \
     --attn=${ATTN}