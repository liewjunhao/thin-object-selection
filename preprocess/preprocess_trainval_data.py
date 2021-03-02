""" Prepare training and validation data (ThinObject-5K). """

import os
import cv2
import math
import shutil
import numpy as np
from PIL import Image
import imageio
from copy import deepcopy

def normalize(img):
    return 255 * (img - img.min()) / (img.max() - img.min())

root_dir = '/Local2T/junhao/data' # Root directory of HRSOD, DIV2K & COCO
fg_root_dir = '../tosnet-v2/dataset/pngimg' # Directory of raw data
bg_root_dir = '../tosnet-v2/dataset/'

# ThinObject-5K directories
fg_dir = os.path.join(fg_root_dir, 'images')
mask_dir = os.path.join(fg_root_dir, 'masks')
comp_dir = 'data/ThinObject5k/images/'
comp_mask_dir = 'data/ThinObject5k/masks/'
os.makedirs(comp_dir, exist_ok=True)
os.makedirs(comp_mask_dir, exist_ok=True)
fg_names = os.path.join(fg_root_dir, 'list', 'trainval.txt')

# BG directories
bg_dir_coco = os.path.join(root_dir, 'mscoco', 'coco', 'images', 'train2014')
bg_dir_div2k = os.path.join(bg_root_dir, 'div2k')
bg_dir_hrsod = os.path.join(bg_root_dir, 'hrsod', 'all_images')
bg_names_coco = os.path.join(root_dir, 'mscoco', 'coco', 'list', 'coco_train2014.txt')
bg_names_div2k = os.path.join(bg_dir_div2k, 'list', 'all_imgs.txt')
bg_names_hrsod = os.path.join(bg_dir_hrsod, '..', 'list', 'all_imgs.txt')

# Read file names
with open(os.path.join(fg_names), 'r') as f:
    fg_ids = f.read().splitlines()
with open(os.path.join(bg_names_coco), 'r') as f:
    bg_coco_ids = f.read().splitlines()
with open(os.path.join(bg_names_div2k), 'r') as f:
    bg_div2k_ids = f.read().splitlines()
with open(os.path.join(bg_names_hrsod), 'r') as f:
    bg_hrsod_ids = f.read().splitlines()

# Print statistics
print('#FG: {}'.format(len(fg_ids)))
print('#BG (COCO): {}'.format(len(bg_coco_ids)))
print('#BG (DIV2K): {}'.format(len(bg_div2k_ids)))
print('#BG (HRSOD): {}'.format(len(bg_hrsod_ids)))
fg_cnt = len(fg_ids)

bg_coco_id, bg_div2k_id, bg_hrsod_id = 0, 0, 0
for i in range(fg_cnt):
    im_name = fg_ids[i]
    fg_path = os.path.join(fg_dir, im_name)
    mask_path = os.path.join(mask_dir, im_name)
    assert(os.path.exists(fg_path))
    assert(os.path.exists(mask_path))

    fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)[:,:,:3][:,:,::-1]
    mask = np.array(Image.open(mask_path))
    fg, mask = np.float32(fg), np.float32(mask)
    # Check the range
    if fg.max() > 255:
        print('**************************************************************')
        print('WARNING: Encounter FG pixel value > 255 ({}). '
                'Perform normalization.'.format(fg.max()))
        print('**************************************************************')
        fg = normalize(fg)
        mask = normalize(mask)
        
    assert(mask.shape == fg.shape[:2])
    # print('Mask value: {}'.format(mask.max()))
    h, w, c = fg.shape
    
    size = np.maximum(h, w)
    if size <= 640:
        # Use COCO as background
        bg_path = os.path.join(bg_dir_coco, bg_coco_ids[bg_coco_id])
        bg_coco_id += 1
    else:
        # Use DIV2K or HRSOD as background
        if bg_div2k_id < len(bg_div2k_ids):
            bg_path = os.path.join(bg_dir_div2k, bg_div2k_ids[bg_div2k_id])
            bg_div2k_id += 1
        else:
            bg_path = os.path.join(bg_dir_hrsod, bg_hrsod_ids[bg_hrsod_id])
            bg_hrsod_id += 1
    
    if ((bg_div2k_id == len(bg_div2k_ids)) and (bg_hrsod_id == len(bg_hrsod_ids))):
        bg_div2k_id = 0
        bg_hrsod_id = 0

    print('ID: {}, Size: {}, Path: {}'.format(i, size, bg_path))
    assert(os.path.exists(bg_path))

    # Read background
    bg = np.array(Image.open(bg_path))
    bg = np.float32(bg)
    if len(bg.shape) == 2:
        bg = np.stack((bg,)*3, -1)
    assert(len(bg.shape) == 3)
    bh, bw, bc = bg.shape

    wratio = float(w) / bw
    hratio = float(h) / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        new_bw = int(bw * ratio + 1.0)
        new_bh = int(bh * ratio + 1.0)
        bg = cv2.resize(bg, (new_bw, new_bh), interpolation=cv2.INTER_LINEAR)
    bg = bg[:h, :w, :]

    # Check the range
    if bg.max() > 255:
        print('**************************************************************')
        print('WARNING: Encounter BG pixel value > 255 ({}). '
                'Perform normalization.'.format(bg.max()))
        print('**************************************************************')
        bg = normalize(bg)

    # Save masks
    comp_mask_save_path = os.path.join(comp_mask_dir, im_name)
    Image.fromarray(np.uint8(mask)).save(comp_mask_save_path)

    # Composite
    assert(bg.shape == fg.shape)
    assert(mask.max() > 1 and mask.max() <= 255)
    assert(fg.max() <= 255)
    assert(bg.max() <= 255)
    mask = mask / 255.
    mask = mask[:, :, np.newaxis]
    comp = fg * mask + bg * (1 - mask)
    assert(comp.max() > 1 and comp.max() <= 255)

    # Save composition results
    img_save_id = im_name[:-4] + '.jpg'
    comp_save_path = os.path.join(comp_dir, img_save_id)
    Image.fromarray(np.uint8(comp)).save(comp_save_path)

    # import pdb; pdb.set_trace()