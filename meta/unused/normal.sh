#!/bin/bash

# --- 配置区 ---
export CUDA_VISIBLE_DEVICES=1
export NCCL_ALGO=tree
CONFIG_FILE="./acc_cfg.yaml"
BASE_OUTPUT_DIR="/data1/dif/nft_output" # 定义基础输出目录

# --- 动态生成目录名 ---
# 使用日期和时间创建一个唯一的实验目录，例如：experiment_20251005_153000
RUN_NAME="experiment_$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_NAME"
LOG_FILE="$OUTPUT_DIR/train.log"

# --- 执行区 ---
echo "本次运行的输出目录为: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "启动训练，日志将输出到: $LOG_FILE"
nohup accelerate launch --config_file "$CONFIG_FILE" train.py \
  output_dir="$OUTPUT_DIR" losses="normal" > "$LOG_FILE" 2>&1 &

echo "训练任务已提交到后台，进程ID: $!，输出目录: $OUTPUT_DIR"