#!/bin/bash
input_dir=$1
output_dir_auto=$2
output_dir_hist=$3
output_dir_reti=$4

# conda init bash

# Create output directories if not exist
mkdir -p $output_dir_auto
mkdir -p $output_dir_hist
mkdir -p $output_dir_reti

# set environment to cv
conda activate cv
python ./Autoencoder/run_model.py $input_dir $output_dir_auto
python ./HistogramEqualization/histeq.py $input_dir $output_dir_hist

conda activate py36
python ./RetinexNet/main.py --use_gpu=1 --gpu_idx=0 --gpu_mem=0.5 --phase=test --test_dir= $input_dir --save_dir= $output_dir_reti --decom=0
