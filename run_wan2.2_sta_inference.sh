#!/bin/bash

# 检查 LOG_PATH 环境变量是否存在且非空
if [ -n "$LOG_PATH" ]; then
    # 如果 LOG_PATH 存在，设置 LOG_DIR 为 ./logs/LOG_PATH
    LOG_DIR="./logs/${LOG_PATH}"
else
    # 如果 LOG_PATH 不存在，使用当前时间创建目录
    CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
    LOG_DIR="./logs/${CURRENT_TIME}"
fi

mkdir -p ${LOG_DIR}
export LOG_DIR="$LOG_DIR"

echo "LOG_DIR: $LOG_DIR"

OUTPUT_VIDEO_NAME="output_video.mp4"
SAMPLE_STEPS=40
PROMP="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

NUM_GPUS=8

export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./examples/inference/basic/basic_wan2_2_i2v.py \
    --ckpt_dir "/wk_dir/Wan_Models/Wan2.2-I2V-A14B-Diffusers" \
    --sample_steps 40 \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --num_gpus ${NUM_GPUS} \
    --save_file ${LOG_DIR}/${OUTPUT_VIDEO_NAME} \
    --prompt "${PROMP}" \
    --base_seed 20 \
    --image /wk_dir/wan2.2_mtt/examples/i2v_input.JPG \
    --sample_steps ${SAMPLE_STEPS} \
    --dit_fsdp \
    # --log_dir ${LOG_DIR} \
    # --redirect 3