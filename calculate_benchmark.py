# collect images from ./output_dir/ and calculate the PSNR and SSIM values by comparing with corresponding images present in ./input_dir/
# save the results in ./benchmark_results.csv

import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

groundTruth_dir = './ground_truth/'
output_dir_ae = './output_dir/Autoencoder/'
output_dir_hist = './output_dir/HistogramEq/'
output_dir_retinex = './output_dir/RetinexNet/'
benchmark_results_ae = './autoencoder_results.csv'
benchmark_results_hist = './histogram_eq_results.csv'
benchmark_results_retinex = './retinex_results.csv'

def calculate_psnr_ssim(groundTruth_dir, output_dir, suffix=None):
    input_images = os.listdir(groundTruth_dir)
    output_images = os.listdir(output_dir)
    results = []
    for img in input_images:
        if suffix == None:
            if img in output_images:
                # print(f"Calculating PSNR and SSIM for {img}")
                input_img = cv2.imread(groundTruth_dir + img)
                output_img = cv2.imread(output_dir + img)
                # resize the ground truth image to match the output image
                input_img = cv2.resize(input_img, (output_img.shape[1], output_img.shape[0]))
                psnr = cv2.PSNR(input_img, output_img)
                ssim_val = ssim(input_img, output_img, multichannel=True, channel_axis=2)
                results.append([img, psnr, ssim_val])
        else:
            augmented_img = img.split('.')[0]+"_S."+img.split('.')[1]
            # print(augmented_img)
            if augmented_img in output_images:
                input_img = cv2.imread(groundTruth_dir + img)
                output_img = cv2.imread(output_dir + augmented_img)
                psnr = cv2.PSNR(input_img, output_img)
                ssim_val = ssim(input_img, output_img, multichannel=True, channel_axis=2)
                results.append([img, psnr, ssim_val])
    return results

results_ae = calculate_psnr_ssim(groundTruth_dir, output_dir_ae)
df = pd.DataFrame(results_ae, columns=['Image', 'PSNR', 'SSIM'])
df.to_csv(benchmark_results_ae, index=False)

print('Benchmark results saved in', benchmark_results_ae)
print(df)

results_hist = calculate_psnr_ssim(groundTruth_dir, output_dir_hist)
df = pd.DataFrame(results_hist, columns=['Image', 'PSNR', 'SSIM'])
df.to_csv(benchmark_results_hist, mode='a', index=False)

print('Benchmark results saved in', benchmark_results_hist)
print(df)

results_retinex = calculate_psnr_ssim(groundTruth_dir, output_dir_retinex, suffix="_S")
df = pd.DataFrame(results_retinex, columns=['Image', 'PSNR', 'SSIM'])
df.to_csv(benchmark_results_retinex, mode='a', index=False)

print('Benchmark results saved in', benchmark_results_retinex)
print(df)
