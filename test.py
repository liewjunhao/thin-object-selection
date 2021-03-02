""" Test TOS-Net. """

import os
import numpy as np
from datetime import datetime
import sys
import argparse
from tqdm import tqdm
import imageio
from collections import OrderedDict

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate, sigmoid
import torch.backends.cudnn as cudnn

import dataloaders.thinobject5k as thinobject5k
import dataloaders.coift as coift
import dataloaders.hrsod as hrsod
from dataloaders import custom_transforms as tr
import dataloaders.helpers as helpers
import networks.tosnet as tosnet

def parse_args():
    parser = argparse.ArgumentParser(description='Test TOS-Net')
    parser.add_argument('--test_set', type=str, default='coift')
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--cfg', type=str, default='weights/tosnet_ours/config.txt')
    parser.add_argument('--weights', type=str, default='weights/tosnet_ours/models/TOSNet_epoch-49.pth')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = helpers.read_cfg(args.cfg)
    for key in cfg.keys():
        print('[{}]: {}'.format(key, cfg[key]))

    device = torch.device('cuda')
    # cudnn.enabled = True
    # cudnn.benchmark = True
    # cudnn.deterministic = True

    # Setup network
    tosnet.lr_size = cfg['lr_size']
    net = tosnet.tosnet_resnet50(
                n_inputs=cfg['num_inputs'],
                n_classes=cfg['num_classes'],
                os=16, pretrained=None)
    print('Loading from snapshot: {}'.format(args.weights))
    net.load_state_dict(torch.load(args.weights, map_location=lambda storage, loc:storage))
    net.to(device)
    net.eval()

    # Setup data transformations
    composed_transforms = [
        tr.IdentityTransform(tr_elems=['gt'], prefix='ori_'),
        tr.CropFromMask(crop_elems=['image', 'gt'], relax=cfg['relax_crop'], 
                zero_pad=cfg['zero_pad_crop'], 
                adaptive_relax=cfg['adaptive_relax'], prefix=''),
        tr.Resize(resize_elems=['image', 'gt', 'void_pixels'],
                min_size=cfg['min_size'], max_size=cfg['max_size']),
        tr.ComputeImageGradient(elem='image'),
        tr.ExtremePoints(sigma=10, pert=0, elem='gt'),
        tr.GaussianTransform(tr_elems=['extreme_points'],
                mask_elem='gt', sigma=10, tr_name='points'),
        tr.FixedResizePoints(resolutions={
                'extreme_points': (cfg['lr_size'], cfg['lr_size'])},
                mask_elem='gt', prefix='lr_'),
        tr.FixedResize(resolutions={
                'image' : (cfg['lr_size'], cfg['lr_size']),
                'gt'    : (cfg['lr_size'], cfg['lr_size']),
                'void_pixels': (cfg['lr_size'], cfg['lr_size'])},
                prefix='lr_'),
        tr.GaussianTransform(tr_elems=['lr_extreme_points'],
                mask_elem='lr_gt', sigma=10, tr_name='lr_points'),
        tr.ToImage(norm_elem=['points', 'image_grad', 'lr_points']),
        tr.ConcatInputs(cat_elems=['lr_image', 'lr_points'], cat_name='concat_lr'),
        tr.ConcatInputs(cat_elems=['image', 'points'], cat_name='concat'),
        tr.ConcatInputs(cat_elems=['image', 'image_grad'], cat_name='grad'),
        tr.ToTensor()]
    composed_transforms_ts = transforms.Compose(composed_transforms)

    # Setup dataset
    if args.test_set == 'thinobject5k_test':
        db = thinobject5k.ThinObject5K(split='test', transform=composed_transforms_ts)
    elif args.test_set == 'coift':
        db = coift.COIFT(split='test', transform=composed_transforms_ts)
    elif args.test_set == 'hrsod':
        db = hrsod.HRSOD(split='test', transform=composed_transforms_ts)
    else:
        raise NotImplementedError
    testloader = DataLoader(db, batch_size=1, shuffle=False, num_workers=4)

    # Create result directories
    if args.result_dir is None:
        save_dir = os.path.join('results', args.test_set)
    else:
        save_dir = args.result_dir
    os.makedirs(save_dir, exist_ok=True)

    print('Testing network')
    with torch.no_grad():
        for ii, sample in enumerate(tqdm(testloader)):

            # Read (image, gt) pairs
            inputs = sample['concat'].to(device)
            inputs_lr = sample['concat_lr'].to(device)
            grads = sample['grad'].to(device)
            metas = sample['meta']

            # Forward pass
            outs = net.forward(inputs, grads, inputs_lr, roi=None)[1]
            assert outs.size()[2:] == inputs.size()[2:]
            output = torch.sigmoid(outs).cpu().numpy().squeeze()

            # Project back to original image space
            relax = sample['meta']['relax'][0].item()
            gt = sample['ori_gt'].numpy().squeeze()
            bbox = helpers.get_bbox(gt, pad=relax, zero_pad=True)
            result = helpers.crop2fullmask(output, bbox, gt, zero_pad=True, relax=relax)
            result = np.uint8(result * 255)

            # Save results
            imageio.imwrite(os.path.join(save_dir, metas['image'][0] + \
                                '-' + metas['object'][0] + '.png'), result)
        
    print('Done testing for dataset: {}'.format(args.test_set))
