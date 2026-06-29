#!/bin/bash
cd /home/jupyter-st126222/Project/rdd2022/src

echo "========================================="
echo "Starting all ablation runs: $(date)"
echo "========================================="

echo "--- [1/3] Label Fraction Ablation ---"
python run.py --ablation fraction --epochs 20 --imgsz 608

echo "--- [2/3] Label Imbalance Ablation ---"
python run.py --ablation imbalance --epochs 20 --imgsz 608

echo "--- [3/3] Anchor Sensitivity Ablation ---"
python run.py --ablation anchor --epochs 20 --imgsz 608

echo "========================================="
echo "All runs complete: $(date)"
echo "========================================="
