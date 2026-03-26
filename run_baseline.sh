#!/usr/bin/env bash

# 7个方法 × 3个数据集（5-fold）
# 方法映射：
# M1 shallow_cnn
# M2 resnet34
# M3 simple_vit
# M4 mobilenetv4_conv_small
# M5 efficientvit_m0
# M6 edith
# M7 ecgxtractor   # 若你想换成 ecgiot，把这个改成 ecgiot

for d in ptb ecgid cybhi; do
  for m in shallow_cnn resnet34 simple_vit mobilenetv4_conv_small efficientvit_m0 edith ecgxtractor; do
    echo "=== dataset=$d, model=$m ==="
    ./run_cv.sh -k 5 --start-fold 0 --end-fold 4 \
      -s baseline=true \
      -s kd=false \
      -s scheme=A \
      -s dataset=$d \
      -s baseline_model=$m
  done
done
