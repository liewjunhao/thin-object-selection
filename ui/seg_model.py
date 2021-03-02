import torch
from torch.nn.functional import interpolate, sigmoid
import dataloaders.helpers as helpers
import cv2
import numpy as np

class SegModel(object):
    def __init__(self, net, device, cfg):
        self.net = net
        self.cfg = cfg
        self.device = device

    def _segment(self, image, extreme_points_ori):
        with torch.no_grad():
            # Crop the image
            h, w = image.shape[:2]
            if self.cfg['adaptive_relax']:
                mean_shape = np.mean(image.shape[:2])
                relax = int(self.cfg['relax_crop'] * mean_shape / 428.)
            else:
                relax = self.cfg['relax_crop']
            bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=relax, zero_pad=self.cfg['zero_pad_crop'])
            crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=self.cfg['zero_pad_crop'])

            # Compute the offsets of extreme points
            bounds = (0, 0, w-1, h-1)
            bbox_valid = (max(bbox[0], bounds[0]),
                            max(bbox[1], bounds[1]),
                            min(bbox[2], bounds[2]),
                            min(bbox[3], bounds[3]))
            if self.cfg['zero_pad_crop']:
                offsets = (-bbox[0], -bbox[1])
            else:
                offsets = (-bbox_valid[0], -bbox_valid[1])
            crop_extreme_points = extreme_points_ori + offsets         

            # Resize
            if (np.minimum(h, w) < self.cfg['min_size']) or (np.maximum(h, w) > self.cfg['max_size']):
                sc1 = self.cfg['min_size'] / np.minimum(h, w)
                sc2 = self.cfg['max_size'] / np.maximum(h, w)
                if sc1 > 1:
                    sc = sc1
                else:
                    sc = np.maximum(sc1, sc2)
                resize_image = cv2.resize(crop_image, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                points = crop_extreme_points * sc
            else:
                resize_image = crop_image
                points = crop_extreme_points
            h2, w2 = resize_image.shape[:2]

            # Compute image gradient using Sobel filter
            img_r = resize_image[:, :, 0]
            img_g = resize_image[:, :, 1]
            img_b = resize_image[:, :, 2]
            grad_r = helpers.imgradient(img_r)[0]
            grad_g = helpers.imgradient(img_g)[0]
            grad_b = helpers.imgradient(img_b)[0]
            image_grad = np.sqrt(grad_r**2 + grad_g**2 + grad_b**2)
            # Normalize to [0,1]
            image_grad = (image_grad - image_grad.min()) / (image_grad.max() - image_grad.min())
            
            # Convert extreme points to Gaussian heatmaps
            heatmap = helpers.gaussian_transform(resize_image, points, sigma=10)
            
            # Resize to a fixed resolution to for global context extraction
            resolution = (self.cfg['lr_size'], self.cfg['lr_size'])
            lr_points = np.array([resolution[0] / w2,  resolution[1] / h2]) * points
            lr_image = helpers.fixed_resize(resize_image, resolution)

            # Convert the extreme points to Gaussian heatmaps
            lr_heatmap = helpers.gaussian_transform(lr_image, lr_points, sigma=10)

            # Normalize inputs
            heatmap = 255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
            lr_heatmap = 255 * (lr_heatmap - lr_heatmap.min()) / (lr_heatmap.max() - lr_heatmap.min() + 1e-10)
            image_grad = 255 * (image_grad - image_grad.min()) / (image_grad.max() - image_grad.min() + 1e-10)

            # Concatenate the inputs (1, H, W, C)
            concat_lr = np.concatenate([lr_image, lr_heatmap[:,:,None]], axis=-1)[None,:].transpose(0, 3, 1, 2)
            concat = np.concatenate([resize_image, heatmap[:,:,None]], axis=-1)[None,:].transpose(0, 3, 1, 2)
            grad = np.concatenate([resize_image, image_grad[:,:,None]], axis=-1)[None,:].transpose(0, 3, 1, 2)

            # Convert to PyTorch tensors
            concat_lr = torch.from_numpy(concat_lr).float().to(self.device)
            concat = torch.from_numpy(concat).float().to(self.device)
            grad = torch.from_numpy(grad).float().to(self.device)

            # Forward pass
            outs = self.net.forward(concat, grad, concat_lr, roi=None)[1]
            output = torch.sigmoid(outs).cpu().numpy().squeeze()

            # Project back to original image space
            result = helpers.crop2fullmask(output, bbox, im_size=image.shape[:2], zero_pad=True, relax=relax)

        return result