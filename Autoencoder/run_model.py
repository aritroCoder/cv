# Environment: cv
import numpy as np
import pandas as pd
import os
import cv2 as cv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader
import argparse

SIZE = 256

def img_to_array(img, data_format=None, dtype=None):
    """Converts a PIL Image instance to a Numpy array.

    Usage:

    ```python
    from PIL import Image
    img_data = np.random.random(size=(100, 100, 3))
    img = tf.keras.utils.array_to_img(img_data)
    array = tf.keras.utils.image.img_to_array(img)
    ```


    Args:
        img: Input PIL Image instance.
        data_format: Image data format, can be either `"channels_first"` or
          `"channels_last"`. None means the global
          setting `tf.keras.backend.image_data_format()` is used (unless you
          changed it, it uses `"channels_last"`). Defaults to `None`.
        dtype: Dtype to use. None makes the global setting
          `tf.keras.backend.floatx()` to be used (unless you changed it, it
          uses `"float32"`). Defaults to `None`.

    Returns:
        A 3D Numpy array.

    Raises:
        ValueError: if invalid `img` or `data_format` is passed.
    """
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x

def load_images(path, augment=True):
    x = []
    image_paths = []
    for imageDir in os.listdir(path):
      img_path = os.path.join(path, imageDir)
      image_paths.append(img_path)
    image_paths.sort()
    for img_path in image_paths:
        img = cv.imread(img_path,1)
        if img is None:
            continue
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        x.append(img_to_array(img, data_format="channels_first"))

        if augment:
            # performing data augmentation by rotating, and flipping the image
            img1 = cv.flip(img,1)
            x.append(img_to_array(img1, data_format="channels_first"))

            img2 = cv.flip(img,-1)
            x.append(img_to_array(img2, data_format="channels_first"))

            img3 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            x.append(img_to_array(img3, data_format="channels_first"))

            img4 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            x.append(img_to_array(img4, data_format="channels_first"))

    return x

class Downsample(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, apply_batch_normalization=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.apply_batch_normalization = apply_batch_normalization
        if apply_batch_normalization:
            self.batch_norm = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.apply_batch_normalization:
            x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, dropout=False):
        super(Upsample, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.dropout = dropout
        if dropout:
            self.dropout_layer = nn.Dropout2d(p=0.1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_transpose(x, output_size=(x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2))
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.leaky_relu(x)
        return x

class Model(nn.Module):
    def __init__(self, kernel=3):
        super(Model, self).__init__()
        self.down1 = Downsample(in_channels=3, filters=128, kernel_size=kernel, apply_batch_normalization=False)
        self.down2 = Downsample(in_channels=128, filters=256, kernel_size=kernel, apply_batch_normalization=False)
        self.down3 = Downsample(in_channels=256, filters=512, kernel_size=kernel, apply_batch_normalization=True)
        self.down4 = Downsample(in_channels=512, filters=512, kernel_size=kernel, apply_batch_normalization=True)

        self.up1 = Upsample(in_channels=512, filters=512, kernel_size=kernel, dropout=True)
        self.up2 = Upsample(in_channels=1024, filters=256, kernel_size=kernel, dropout=True)
        self.up3 = Upsample(in_channels=512, filters=128, kernel_size=kernel, dropout=True)
        self.up4 = Upsample(in_channels=256, filters=3, kernel_size=kernel, dropout=True)

        self.final_conv = nn.Conv2d(6, 3, kernel_size=(2, 2), stride=1, padding="same")

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        r = self.down4(d3)
        
        u1 = self.up1(r)
        u1 = torch.cat([u1, d3], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x], dim=1)

        output = self.final_conv(u4)
        
        return output

class LLEDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Model(kernel=5)
model= nn.DataParallel(model) # Uncomment this line if you want to use multiple GPUs
model.to(device)

model.load_state_dict(torch.load('/DATA/sujit_2021cs35/cv/Autoencoder/model_weights.pth'))

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Directory with images to be enhanced')
parser.add_argument('output', type=str, help='Directory to save enhanced images')

args = parser.parse_args()
test_x = load_images(args.input, augment=False)

for i in range(len(test_x)):
    input_img = torch.tensor(test_x[i]).unsqueeze(0).to(device)
    yhat = model(input_img).cpu().detach().numpy()
    predicted = np.clip(yhat,0.0,1.0)
    op_path = os.path.join(args.output, f'{i}.png')
    plt.imsave(op_path, predicted.squeeze(0).transpose(1, 2, 0))

# python ./Autoencoder/run_model.py /DATA/sujit_2021cs35/cv/Autoencoder/test_img_resized /DATA/sujit_2021cs35/cv/Autoencoder/output_img
