# cv environment
import cv2
import os
import argparse

def HistEq(img_dir, output_dir):
    for img_n in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_n)
        img = cv2.imread(img_path, 1)
        R, G, B = cv2.split(img)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        output_path = os.path.join(output_dir, img_n)
        cv2.imwrite(output_path, equ)

# args: directory with 256x256 images to be enhanced
# python ./HistogramEqualization/histeq.py /DATA/sujit_2021cs35/cv/HistogramEqualization/test_img_resized /DATA/sujit_2021cs35/cv/HistogramEqualization/output_img
parser = argparse.ArgumentParser(description='Histogram Equalization')
parser.add_argument('img_dir', type=str, help='Directory with images to be enhanced')
parser.add_argument('output_dir', type=str, help='Directory to save enhanced images')
args = parser.parse_args()
img_dir = args.img_dir
output_dir = args.output_dir
HistEq(img_dir, output_dir)
