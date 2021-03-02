""" Evaluation. """
import os
from collections import OrderedDict
import numpy as np
import sys
import argparse
import tqdm
from PIL import Image
import cv2
from torchvision import transforms

# import dataloaders.hrsod as hrsod
import dataloaders.thinobject5k as thinobject5k
import dataloaders.coift as coift
import dataloaders.hrsod as hrsod
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
import dataloaders.helpers as helpers
from evaluations import jaccard, f_boundary

gt_thin_root_dir = 'data/thin_regions/'
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating...')
    parser.add_argument('--test_set', type=str, default='coift')
    parser.add_argument('--result_dir', type=str, default='results/coift/')
    parser.add_argument('--thres', type=float, default=0.5)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Setup dataset
    if args.test_set == 'coift':
        db = coift.COIFT(split='test')
    elif args.test_set == 'thinobject5k_test':
        db = thinobject5k.ThinObject5K(split='test')
    elif args.test_set == 'hrsod':
        db = hrsod.HRSOD(split='test')
    else:
        raise NotImplementedError
    testloader = DataLoader(db, batch_size=1, shuffle=False, num_workers=4)

    # Initialize
    all_iou = np.zeros(len(testloader))
    all_iou_thin = np.zeros(len(testloader))
    all_f_boundary = np.zeros(len(testloader))

    for ii, sample in enumerate(tqdm.tqdm(testloader)):

        # Read ground truth
        gt = sample['gt'].numpy().squeeze()
        metas = sample['meta']

        # Read segmentation mask
        filename = os.path.join(args.result_dir, metas['image'][0] + '-' + \
                    metas['object'][0] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32) / 255.
        mask = np.float32(mask > args.thres)
        assert gt.shape == mask.shape

        # Read thin regions (for evaluation of IoU_thin)
        filename = os.path.join(gt_thin_root_dir, args.test_set, 'eval_mask',
                    metas['image'][0] + '-' + metas['object'][0] + '.png')
        gt_thin = np.array(Image.open(filename)).astype(np.float32)
        assert gt.shape == gt_thin.shape

        # Evaluate IoU
        all_iou[ii] = jaccard.jaccard(gt, mask)
        # Evaluate IoU_thin
        void_thin = np.float32(gt_thin == 255)
        all_iou_thin[ii] = jaccard.jaccard(gt, mask, void_thin)
        # Evaluate F-boundary
        all_f_boundary[ii] = f_boundary.db_eval_boundary(mask, gt)

    # Compute average stats
    mean_iou = all_iou.mean()
    print('IoU: {}'.format(mean_iou))
    mean_iou_thin = all_iou_thin.mean()
    print('IoU_thin: {}'.format(mean_iou_thin))
    mean_f_boundary = all_f_boundary.mean()
    print('F-boundary: {}'.format(mean_f_boundary))
