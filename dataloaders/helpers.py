""" Helper functions. """
import os
import cv2
import torch
import numpy as np
from scipy import ndimage
import re
import itertools
from copy import deepcopy
import random
from skimage.feature import peak_local_max
import scipy.sparse.linalg as sla
from collections import OrderedDict

def preprocess_image(image):
    """ Pre-process image such that it is 3-channel and in the range of [0,255]. """
    if image.ndim == 2:
        image = np.uint8(np.stack((deepcopy(image),)*3, -1))
    if image.max() < 1:
        image = image * 255
    return image

def mask_image(image, mask, color=[255,0,0], alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    image = preprocess_image(image)
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    contours = cv2.findContours(np.uint8(deepcopy(mask)), cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(out, contours[0], -1, (0.0, 0.0, 0.0), 1)
    return out

def read_cfg(cfg_file):
    """ Read configuration file. """
    with open(cfg_file, 'r') as f:
        lines = f.read().splitlines()
    _cfg = OrderedDict()
    for line in lines:
        line = line.split(':')
        if len(line) == 2:
            arg, val = line[0], line[1]
            _cfg[arg] = val

    # Load the corresponding arguments
    cfg = OrderedDict()
    cfg['backbone'] = _cfg['backbone']
    cfg['num_inputs'] = int(_cfg['num_inputs'])
    cfg['num_classes'] = int(_cfg['num_classes'])
    cfg['lr_size'] = int(_cfg['lr_size'])
    cfg['min_size'] = int(_cfg['min_size'])
    cfg['max_size'] = int(_cfg['max_size'])
    cfg['relax_crop'] = int(_cfg['relax_crop'])
    cfg['zero_pad_crop'] = True if _cfg['zero_pad_crop'] == 'True' else False
    cfg['adaptive_relax'] = True if _cfg['adaptive_relax'] == 'True' else False
    return cfg

def imgradient(mask):
    """ Equivalent to 'imgradient' in MATLAB with mode='sobel'
    Reference: https://stackoverflow.com/a/47835313
    """
    sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    return magnitude, angle

def extract_edge(mask, method='dist_transf'):
    """ Extract the object boundaries."""
    if mask.max() == 0:
        return np.zeros_like(mask)
    if method == 'imgradient':
        edge = imgradient(mask)[0]
    elif method == 'dist_transf':
        dt = ndimage.distance_transform_edt(mask)
        edge = np.logical_and(dt <= 1, mask)
    else:
        raise NotImplementedError
    return (edge > 0).astype(np.float32)

def gaussian_transform(img, labels, sigma=10):
    """ Perform gaussian transformation using distance transformation.
    Args:
        img: The original color image
        labels: Label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
        sigma: Sigma of the Gaussian.
    """
    h, w = img.shape[:2]
    if labels is None:
        heatmap = np.zeros((h, w))
    else:
        points = np.zeros((h, w))
        labels = np.round(labels).astype(np.int)
        points[labels[:,1], labels[:,0]] = 1
        heatmap = ndimage.distance_transform_edt(np.logical_not(points))
        heatmap = np.exp(-heatmap / sigma)
    return heatmap

"""For training only."""
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))

def adjust_lr_poly(optimizer, base_lr, i_iter, max_iter):
    """adjust learning rate using poly scheduler."""
    lr = lr_poly(base_lr, i_iter, max_iter, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 10 * lr
    return lr


""" The following functions are taken from DEXTR (with some minor modifications).
Reference: https://github.com/scaelles/DEXTR-PyTorch
"""
def crop_from_mask(img, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])

    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad)

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad)

    return crop

def get_bbox(mask, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max

def crop_from_bbox(img, bbox, zero_pad=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    img = np.squeeze(img)
    if img.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
            img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

    return crop

def fixed_resize(sample, resolution, flagval=None):
    """Note that the default cv2.INTER_CUBIC is replaced with cv2.INTER_LINEAR."""
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_LINEAR #cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample

def fixed_extreme_points(mask):
    def find_point(idx, idy, ids):
        num_points = len(ids[0])
        sel_id = ids[0][num_points//2]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coodinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_y == np.min(inds_y))), # top
                     find_point(inds_x, inds_y, np.where(inds_x == np.min(inds_x))), # left
                     find_point(inds_x, inds_y, np.where(inds_y == np.max(inds_y))), # bottom
                     find_point(inds_x, inds_y, np.where(inds_x == np.max(inds_x))), # right
                     ])

def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)), # top
                     find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)), # left
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert)), # bottom
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)), # right
                     ])

def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """Make the ground-truth for landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt

def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  interpolation=cv2.INTER_LINEAR, scikit=False, rm_invalid=True):
    """Note that the default cv2.INTER_CUBIC is replaced with cv2.INTER_LINEAR."""
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borders of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()

def natural_keys(text):
    """ alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split('(\d+)', text) ]