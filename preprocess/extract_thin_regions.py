""" Extract thin regions for evaluation of IoU_thin. """

import os
import sys
import random
import numpy as np
import imageio
import argparse
from tqdm import tqdm
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage import measure
import skfmm

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dataloaders.thinobject5k as thinobject5k
import dataloaders.coift as coift
import dataloaders.hrsod as hrsod
import dataloaders.custom_transforms as tr
import dataloaders.helpers as helpers

def parse_args():
    parser = argparse.ArgumentParser(description='Extract thin regions for evaluation...')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='data/thin_regions/')
    args = parser.parse_args()
    return args

def pad_image(img, pad_val=0):
    """ Pad the border of the image with pad_val. """
    h, w = img.shape[:2]
    pad = np.ones((h+2, w+2)) * pad_val
    pad[1:-1, 1:-1] = img.copy()
    return pad

def distance_transform_pad(img):
    """ Apply padding before running ndimage.distance_transform_edt(). """
    pad = pad_image(img)
    dt = ndimage.distance_transform_edt(pad)
    dt = dt[1:-1, 1:-1]
    return dt

def extract_thin_regions_from_mask(mask):
    """ Extract thin regions from a binary mask. """
    assert ((mask == 0) | (mask == 1)).all()
    h, w = mask.shape[:2]
    ys, xs = np.where(mask == 1)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    box_h, box_w = (y2 - y1 + 1), (x2 - x1 + 1)
    max_side = np.maximum(box_h, box_w)
    thresh = 10. * max_side / 300

    # Separate into multiple connected components
    mask_cc = measure.label(mask, background=0)

    # Compute distance to the object boundaries
    dist = distance_transform_pad(mask)
    # Extract the local peak
    coords = peak_local_max(dist)
    dist_vals = dist[coords[:,0], coords[:,1]]

    # The pixel is classified as "thin" if its distance to the closest 
    # boundary is less than "thresh"
    coords = coords[dist_vals<thresh ,:]
    thin = np.zeros_like(mask)
    thin[coords[:,0], coords[:,1]] = 1

    # When there is no thin part
    if thin.sum() == 0:
        return np.zeros_like(mask), thresh

    # Extract the regions that are close to these "thin" pixels
    # However, these regions also contain some non-thin pixels
    # Note that we use geodesic dist to avoid "cross-boundaries" grouping
    # Problem: cannot link to disconnected component
    if len(np.unique(mask_cc)) == 2:
        m = np.ma.masked_array(ndimage.distance_transform_edt(np.logical_not(thin)), 
                    np.logical_not(mask))
        thin_dist = skfmm.distance(m).data
        thin_region = np.logical_and((thin_dist<thresh), mask)
    else:
        # Run geodesic distance separately on each component before merging
        thin_region = np.zeros_like(mask)
        for label in np.unique(mask_cc):
            if label == 0:
                continue
            _mask = mask_cc == label
            _thin = np.logical_and(_mask, thin)
            if _thin.sum() == 0:
                continue
            m = np.ma.masked_array(ndimage.distance_transform_edt(np.logical_not(_thin)), 
                    np.logical_not(_mask))
            thin_dist = skfmm.distance(m).data
            thin_region = np.logical_or(thin_region, 
                                np.logical_and((thin_dist<thresh), _mask))

    # Perform post-processing by looking for pixels that are close to 
    # "non-thin" pixels. Removing this from ground truth produce our final
    # thin regions.
    non_thin_region = mask - thin_region
    if non_thin_region.sum() == 0:
        # The entire object is made up of thin parts
        return thin_region, thresh
    else:
        if len(np.unique(mask_cc)) == 2:
            m = np.ma.masked_array(ndimage.distance_transform_edt(np.logical_not(non_thin_region)), 
                        np.logical_not(mask))
            non_thin_dist = skfmm.distance(m).data
            non_thin = np.logical_and(non_thin_dist < thresh, mask)
        else:
            non_thin = np.zeros_like(mask)
            for label in np.unique(mask_cc):
                if label == 0:
                    continue
                _mask = mask_cc == label
                _non_thin = np.logical_and(_mask, non_thin_region)
                if _non_thin.sum() == 0:
                    continue
                m = np.ma.masked_array(ndimage.distance_transform_edt(np.logical_not(_non_thin)), 
                            np.logical_not(_mask))
                non_thin_dist = skfmm.distance(m).data
                non_thin = np.logical_or(non_thin, 
                                np.logical_and(non_thin_dist < thresh, _mask))
            if non_thin.sum() == 0:
                import pdb; pdb.set_trace()

    thin_region_pp = mask - non_thin

    return thin_region_pp, thresh

def extract_evaluation_band(mask, thin_region, thresh, k=1):
    """ Extract a thin strip around BG that surrounds the thin pixels for 
    evaluation (IoU_thin). """

    # If there is no thin part, the full image is "void" label
    if thin_region.sum() == 0:
        return np.ones_like(mask) * 255
    
    # Extract the regions that are close to these "thin" pixels
    non_void = ndimage.distance_transform_edt(np.logical_not(thin_region))
    non_void = non_void < k*thresh
    # Exclude non-thin foreground pixels
    non_thin_fg = mask - thin_region
    non_void[non_thin_fg == 1] = 0

    # Construct the evaluation band
    eval_mask = mask.copy()
    eval_mask[non_void == 0] = 255

    return eval_mask

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'coift':
        db = coift.COIFT(split='test', transform=None)
    elif args.dataset == 'thinobject5k_val':
        db = thinobject5k.ThinObject5K(split='val', transform=None)
    elif args.dataset == 'thinobject5k_test':
        db = thinobject5k.ThinObject5K(split='test', transform=None)
    elif args.dataset == 'thinobject5k_train':
        db = thinobject5k.ThinObject5K(split='train', transform=None)
    elif args.dataset == 'hrsod':
        db = hrsod.HRSOD(split='test', transform=None)
    else:
        raise NotImplementedError
    testloader = DataLoader(db, batch_size=1, shuffle=False, num_workers=4)

    # Create save directories
    save_dir = os.path.join(args.save_path, args.dataset)
    save_thin_dir = os.path.join(save_dir, 'gt_thin')
    save_eval_dir = os.path.join(save_dir, 'eval_mask')
    os.makedirs(save_thin_dir, exist_ok=True)
    os.makedirs(save_eval_dir, exist_ok=True)

    for ii, sample in enumerate(tqdm(testloader)):

        mask = sample['gt'].numpy().squeeze()
        metas = sample['meta']
        thin_region, thresh = extract_thin_regions_from_mask(mask)

        # Save thin_regions
        thin = np.uint8(255 * thin_region)
        name = metas['image'][0] + '-' + metas['object'][0] + '.png'
        imageio.imwrite(os.path.join(save_thin_dir, name), thin)

        # Prepare evaluation mask for evaluation of IoU_thin
        eval_mask = extract_evaluation_band(mask, thin_region, thresh, k=1)
        imageio.imwrite(os.path.join(save_eval_dir, name), np.uint8(eval_mask))

    print('Done processing for dataset: {}'.format(args.dataset))