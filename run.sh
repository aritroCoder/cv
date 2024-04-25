#!/bin/bash
input_dir=$1
output_dir_auto=$2
output_dir_hist=$3
output_dir_reti=$4

workdir=$(pwd)

# if no CLI arguments are provided, use default values
if [ -z "$input_dir" ]; then
    input_dir="/DATA/sujit_2021cs35/cv/input_dir"
fi

if [ -z "$output_dir_auto" ]; then
    output_dir_auto="/DATA/sujit_2021cs35/cv/output_dir/Autoencoder"
    # clear the directory contents but not the directory
    rm -rf $output_dir_auto/*
fi

if [ -z "$output_dir_hist" ]; then
    output_dir_hist="/DATA/sujit_2021cs35/cv/output_dir/HistogramEq"
    rm -rf $output_dir_hist/*
fi

if [ -z "$output_dir_reti" ]; then
    output_dir_reti="/DATA/sujit_2021cs35/cv/output_dir/RetinexNet"
    rm -rf $output_dir_reti/*
fi

# Create output directories if not exist
mkdir -p $output_dir_auto
mkdir -p $output_dir_hist
mkdir -p $output_dir_reti

# set environment to cv
source activate cv
echo "Run: python ./Autoencoder/run_model.py "+$input_dir" "+" "$output_dir_auto
python ./Autoencoder/run_model.py $input_dir $output_dir_auto
echo "Run: python ./HistogramEqualization/histeq.py "+$input_dir" "+" "$output_dir_hist
python ./HistogramEqualization/histeq.py $input_dir $output_dir_hist

source activate py36
cd RetinexNet
echo "Run: python ./RetinexNet/main.py --use_gpu=1 --gpu_idx=0 --gpu_mem=0.5 --phase=test --test_dir= $input_dir --save_dir= $output_dir_reti --decom=0"
python ./main.py --use_gpu=1 --gpu_idx=0 --gpu_mem=0.5 --phase=test --test_dir=$input_dir --save_dir=$output_dir_reti --decom=0
cd $workdir

# deactivate the environment
source deactivate

echo "Output directories are created at: "
echo $output_dir_auto
echo $output_dir_hist
echo $output_dir_reti

echo "Done"
