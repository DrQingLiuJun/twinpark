#!/bin/bash
# Script to run MPPI control node with PyTorch environment in 'mppi' conda environment

# 激活 mppi 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mppi

# 运行 MPPI 控制节点
exec python3 "$(dirname "$0")/hybrid_Astar_mppi_control.py" "$@"
