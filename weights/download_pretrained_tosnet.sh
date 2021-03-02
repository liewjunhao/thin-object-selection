#!/bin/bash

# Download pre-trained weights
gdown https://drive.google.com/uc?id=1L_H5V8XjLwiPMrb2B4eHyBdPzoEThwz-
gdown https://drive.google.com/uc?id=1kxsQ7ZJJBcFVJ0q73zThHwXjdox51Gw2

# Unzip and clean
unzip tosnet_ours.zip
rm tosnet_ours.zip
