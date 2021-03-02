""" Custom transformations. """
import os, sys
import cv2
import torch
import numpy as np
from scipy import ndimage
from copy import deepcopy
import dataloaders.helpers as helpers
import random

class IdentityTransform(object):
    """ This function is simply used to rename the keys."""
    def __init__(self, tr_elems=[], tr_names=[], prefix='', suffix='', rm=''):
        self.tr_elems = tr_elems
        self.prefix = prefix
        self.suffix = suffix
        self.rm = rm

    def __call__(self, sample):
        for elem in self.tr_elems:
            sample[self.prefix + elem.replace(self.rm, '') + self.suffix] = deepcopy(sample[elem])
        return sample
    
    def __str__(self):
        return 'IdentityTransform(tr_elems=' + str(self.tr_elems) + ')'


class Resize(object):
    """ Resize the inputs such that the shortest side is at least min_size and
    the longer side doesn't exceed max_size.
    """
    def __init__(self, resize_elems=[], min_size=512, max_size=1080, 
                 flagvals=None, prefix=''):
        self.resize_elems = resize_elems
        self.min_size = min_size
        self.max_size = max_size
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert len(self.resize_elems) == len(self.flagvals)
        self.prefix = prefix

    def __call__(self, sample):
        tmp = sample[self.resize_elems[0]]
        h, w = tmp.shape[:2]
        if (np.minimum(h, w) >= self.min_size) and (np.maximum(h, w) <= self.max_size):
            return sample
        else:
            sc1 = self.min_size / np.minimum(h, w)
            sc2 = self.max_size / np.maximum(h, w)
            if sc1 > 1:
                sc = sc1
            else:
                sc = np.maximum(sc1, sc2)

        for ii, elem in enumerate(self.resize_elems):
            tmp = sample[elem]
            if self.flagvals is not None:
                if self.flagvals[ii] == 'cubic':
                    flagval = cv2.INTER_CUBIC
                elif self.flagvals[ii] == 'linear':
                    flagval = cv2.INTER_LINEAR
                elif self.flagvals[ii] == 'nearest':
                    flagval = cv2.INTER_NEAREST
                else:
                    raise NotImplementedError
            else:
                if ((tmp == 0) | (tmp == 1)).all():
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_LINEAR
            tmp = cv2.resize(tmp, (0, 0), fx=sc, fy=sc, interpolation=flagval)
            sample[self.prefix+elem] = tmp

        return sample

    def __str__(self):
        return 'Resize(resize_elems=' + str(self.resize_elems) + \
                ', min_size=' + str(self.min_size) + \
                ', max_size=' + str(self.max_size) + \
                ', flagvals=' + str(self.flagvals) + \
                ', prefix=' + self.prefix + ')'

class ComputeImageGradient(object):
    """ Compute image gradient. """
    def __init__(self, elem, method='sobel'):
        self.elem = elem
        self.method = method

    def __call__(self, sample):
        img = sample[self.elem]
        assert img.ndim == 3 and img.shape[-1] == 3
        if self.method == 'sobel':
            # Apply sobel filter to compute image gradient
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            grad_r = helpers.imgradient(img_r)[0]
            grad_g = helpers.imgradient(img_g)[0]
            grad_b = helpers.imgradient(img_b)[0]
            img_grad = np.sqrt(grad_r**2 + grad_g**2 + grad_b**2)
            # Normalize to [0,1]
            img_grad = (img_grad - img_grad.min()) / (img_grad.max() - img_grad.min())
        else:
            raise NotImplementedError

        sample[self.elem+'_grad'] = img_grad
        return sample

    def __str__(self):
        return 'ComputeImageGradient(elem=' + self.elem + \
                ', method=' + self.method + ')'

class GaussianTransform(object):
    """ Generate a gaussian heatmap for the clicks. """
    def __init__(self, tr_elems=['extreme_points'], mask_elem='gt',
                sigma=10, tr_name='clicks', return_pos=False, approx=True):
        self.tr_elems = tr_elems
        self.mask_elem = mask_elem
        self.sigma = sigma
        self.tr_name = tr_name
        self.return_pos = return_pos # return the binary position of each click
        self.approx = approx # approximate the gaussian transform with distance function

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        h, w = _target.shape[:2]

        heatmap = np.zeros((h, w))
        pos = np.zeros((h, w))
        for elem in self.tr_elems:
            _points = sample[elem]
            if _points is not None:
                if self.approx:
                    heatmap = np.maximum(heatmap, helpers.gaussian_transform(_target, 
                        _points, sigma=self.sigma)) # faster!
                else:
                    heatmap = np.maximum(heatmap, helpers.make_gt(_target, 
                        _points, sigma=self.sigma, one_mask_per_point=False))
                    
                # Return binary positions
                if self.return_pos:
                    _points = _points.astype(int)
                    pos[_points[:,1], _points[:,0]] = 1
        sample[self.tr_name] = heatmap
        if self.return_pos:
            sample[self.tr_name+'_pos'] = pos

        return sample

    def __str__(self):
        return 'GaussianTransform(tr_elems=' + str(self.tr_elems) + \
                ', mask_elem=' + self.mask_elem + \
                ', sigma=' + str(self.sigma) + \
                ', tr_name=' + self.tr_name + ')'

class FixedResizePoints(object):
    """ Resize the clicks to a fixed resolution (disregard aspect ratio)."""
    def __init__(self, resolutions, mask_elem, prefix=''):
        self.resolutions = resolutions
        self.mask_elem = mask_elem
        self.prefix = prefix

    def __call__(self, sample):
        _target = sample[self.mask_elem]
        h, w = _target.shape[:2]

        for elem in self.resolutions.keys():
            points = sample[elem]
            if points is None:
                sample[self.prefix+elem] = None
            else:
                resolution = self.resolutions[elem]
                points = np.array([resolution[0] / w,  resolution[1] / h]) * points
                sample[self.prefix+elem] = points
        return sample

    def __str__(self):
        return 'FixedResizePoints(resolutions=' + str(self.resolutions) + \
                ', mask_elem=' + self.mask_elem + \
                ', prefix=' + self.prefix + ')'

"""For training only."""
class RandomCrop(object):
    """ Randomly crop patches for training. """
    def __init__(self, num_thin, num_non_thin, crop_size, prefix='crop_',
                 thin_elem='thin', crop_elems=[]):
        
        self.num_thin = num_thin # number of patches containing thin parts
        self.num_non_thin = num_non_thin
        self.crop_size = crop_size
        self.thin_elem = thin_elem
        self.crop_elems = crop_elems
        self.prefix = prefix

    def __call__(self, sample):
        _target = sample[self.thin_elem]
        h, w = _target.shape[:2]

        ys, xs = np.where(_target == 1)
        num_thin = np.minimum(len(ys), self.num_thin)
        num_non_thin = self.num_non_thin + (self.num_thin - num_thin)
        if num_thin != 0:
            inds = random.sample(range(0, len(ys)), num_thin)

        # Initialize
        for elem in self.crop_elems:
            tmp = sample[elem]
            if tmp.ndim == 2:
                sample[self.prefix + elem] = np.zeros((num_thin + num_non_thin, 
                            self.crop_size, self.crop_size, 1))
            elif tmp.ndim == 3:
                sample[self.prefix + elem] = np.zeros((num_thin + num_non_thin,
                            self.crop_size, self.crop_size, tmp.shape[-1]))

        rois = []
        for i in range(num_thin + num_non_thin):
            if i < num_thin:
                y, x = ys[inds[i]], xs[inds[i]]
            else:
                y = random.randint(0, h-1)
                x = random.randint(0, w-1)

            # Randomly sample a patch covering the sampled location
            xmin = np.maximum(x - self.crop_size, 0)
            xmax = np.minimum(x, w - self.crop_size)
            x1 = random.randint(xmin, xmax-1) if xmin != xmax else xmin
            ymin = np.maximum(y - self.crop_size, 0)
            ymax = np.minimum(y, h - self.crop_size)
            y1 = random.randint(ymin, ymax-1) if ymin != ymax else ymin
            x2 = x1 + self.crop_size
            y2 = y1 + self.crop_size
            assert (x1 >= 0) and (y1 >= 0)
            assert (x2 <= w) and (y2 <= h)
            rois.append(np.array([[x1, y1, x2, y2]]))

            for elem in self.crop_elems:
                tmp = sample[elem]
                if tmp.ndim == 2:
                    sample[self.prefix+elem][i, :, :, 0] = deepcopy(tmp[y1:y2, x1:x2])
                elif tmp.ndim == 3:
                    sample[self.prefix+elem][i, :] = deepcopy(tmp[y1:y2, x1:x2, :])
        sample['rois'] = np.concatenate(rois, axis=0)

        return sample

    def __str__(self):
        return 'RandomCrop'

class MatchROIs(object):
    def __init__(self, crop_elem='', resolution=416):
        self.crop_elem = crop_elem
        self.resolution = resolution

    def __call__(self, sample):
        crop = sample[self.crop_elem]
        rois = sample['rois']
        sample['rois_ori'] = deepcopy(rois)

        scales = (self.resolution / crop.shape[1], self.resolution / crop.shape[0])
        rois = rois * np.array([scales + scales])
        sample['rois'] = rois
        
        return sample

    def __str__(self):
        return 'MatchROIs(crop_elem=' + self.crop_elem + \
                ', resolution=' + str(self.resolution) + ')'

class ExtractEdge(object):
    """ Extract edges from mask. """
    def __init__(self, mask_elems=[], edge_method='imgradient'):
        self.mask_elems = mask_elems
        self.edge_method = edge_method # 'imgradient'|'dist_transf'

    def __call__(self, sample):
        def isbinary(mask):
            return ((mask==0) | (mask==1)).all()

        for elem in self.mask_elems:
            mask = deepcopy(sample[elem])
            
            if mask.ndim == 4: # (N, H, W, 1)
                edges = []
                for kk in range(mask.shape[0]):
                    tmp = mask[kk, :, :, 0]
                    if not isbinary(tmp):
                        tmp = np.float32(tmp > 0.5)
                    if tmp.max() == 0:
                        edge = np.zeros_like(tmp)
                    else:
                        edge = helpers.extract_edge(tmp, method=self.edge_method)
                    edges.append(edge[np.newaxis, :, :, np.newaxis])
                edges = np.concatenate((edges), axis=0)

            elif mask.ndim == 2: # (H, W)
                if not isbinary(mask):
                    mask = (mask > 0.5).astype(np.float32)
                if mask.max() == 0:
                    edges = np.zeros_like(mask)
                else:
                    edges = helpers.extract_edge(mask, method=self.edge_method)
            else:
                raise NotImplementedError
                
            assert edges.shape == mask.shape
            sample[elem + '_edge'] = edges

        return sample
    
    def __str__(self):
        return 'ExtractEdge(mask_elems=' + str(self.mask_elems) + ')'

class RemoveElements(object):
    """ Remove unwanted elements in the sample. """
    def __init__(self, rm_elems=[]):
        self.rm_elems = rm_elems

    def __call__(self, sample):
        for elem in self.rm_elems:
            if elem in sample.keys():
                del sample[elem]
        return sample

    def __str__(self):
        return 'RemoveElements(rm_elems' + str(self.rm_elems) + ')'


""" The following functions are taken from DEXTR (with some minor modifications).
Reference: https://github.com/scaelles/DEXTR-PyTorch
"""
class RandomHorizontalFlip(object):
    """Random horizontal flipping for data augmentation."""
    def __call__(self, sample):
        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp
        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'

class ConcatInputs(object):
    """Concatenate inputs."""
    def __init__(self, cat_elems=[], cat_name='concat'):
        self.cat_elems = cat_elems
        self.cat_name = cat_name

    def __call__(self, sample):
        out = sample[self.cat_elems[0]]
        
        if out.ndim == 3:
            # format: H x W x C
            for elem in self.cat_elems[1:]:
                tmp = sample[elem]
                assert sample[self.cat_elems[0]].shape[:2] == tmp.shape[:2]
                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis] # add missing 3rd dimension
                out = np.concatenate((out, tmp), axis=2)
        elif out.ndim == 4:
            # format: N x H x W x C
            for elem in self.cat_elems[1:]:
                tmp = sample[elem]
                assert sample[self.cat_elems[0]].shape[:3] == tmp.shape[:3]
                out = np.concatenate((out, tmp), axis=-1)
        else:
            raise NotImplementedError

        sample[self.cat_name] = out
        return sample

    def __str__(self):
        return 'ConcatInputs(cat_elems=' + str(self.cat_elems) + \
                ', cat_name=' + self.cat_name + ')'

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, excludes=[]):
        self.excludes = excludes

    def __call__(self, sample):
        for elem in sample.keys():
            if 'meta' in elem:
                continue
            if elem in self.excludes:
                continue
            
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            if tmp.ndim == 3:
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                tmp = tmp.transpose((2, 0, 1))
            elif tmp.ndim == 4:
                # numpy image: N x H x W x C
                # torch image: N x C x H x W
                tmp = tmp.transpose((0, 3, 1, 2))
            else:
                raise NotImplementedError

            # sample[elem] = torch.from_numpy(tmp).float()
            # to deal with the error: some of strides of a given numpy array are negative
            # https://discuss.pytorch.org/t/error-in-totensor-transformation/4834
            sample[elem] = torch.from_numpy(tmp.copy()).float() # change
            # print('{}:{}'.format(elem, sample[elem].size()))

        return sample

    def __str__(self):
        return 'ToTensor'

class CropFromMask(object):
    """Returns image cropped in bounding box from a given mask.
    Allow adaptive relaxation of bbox according to the image resolution.
    """
    def __init__(self, crop_elems=('image', 'gt'),
                 mask_elem='gt',
                 relax=0,
                 zero_pad=False,
                 adaptive_relax=False,
                 prefix='crop_'):

        self.crop_elems = crop_elems
        self.mask_elem = mask_elem
        self.relax = relax
        self.zero_pad = zero_pad
        self.adaptive_relax = adaptive_relax
        self.prefix = prefix

    def __call__(self, sample):
        _target = sample[self.mask_elem]

        # enable adaptive calculation of bbox relaxation
        if self.adaptive_relax:
            mean_shape = np.mean(_target.shape[:2])
            # 428 is the average size in PASCAL trainaug
            relax = int(self.relax * mean_shape / 428.)
        else:
            relax = self.relax
        sample['meta']['relax'] = relax

        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            _img = sample[elem]
            _crop = []
            if self.mask_elem == elem:
                if _img.ndim == 2:
                    _img = np.expand_dims(_img, axis=-1)
                for k in range(0, _target.shape[-1]):
                    _tmp_img = _img[..., k]
                    _tmp_target = _target[..., k]
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                    else:
                        _crop.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=relax, zero_pad=self.zero_pad))
            else:
                for k in range(0, _target.shape[-1]):
                    if np.max(_target[..., k]) == 0:
                        _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                    else:
                        _tmp_target = _target[..., k]
                        _crop.append(helpers.crop_from_mask(_img, _tmp_target, relax=relax, zero_pad=self.zero_pad))
            if len(_crop) == 1:
                sample[self.prefix + elem] = _crop[0]
            else:
                sample[self.prefix + elem] = _crop
        return sample

    def __str__(self):
        return 'CropFromMask:(crop_elems='+str(self.crop_elems)+', mask_elem='+str(self.mask_elem)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+\
                ',adaptive_relax='+str(self.adaptive_relax)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None, prefix=''):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))
        self.prefix = prefix

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        for elem in elems:

            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[self.prefix+elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[self.prefix+elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[self.prefix+elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[self.prefix+elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[self.prefix+elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[self.prefix+elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                # del sample[elem]
                pass

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)

class ExtremePoints(object):
    """Returns the four extreme points (left, right, top, bottom) (with some random perturbation) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    ***The generation of heatmap is moved to tr.GaussianTransform() instead.***
    """
    def __init__(self, sigma=10, pert=0, elem='gt', prefix=''):
        self.sigma = sigma
        self.pert = pert
        self.elem = elem
        self.prefix = prefix

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            # sample[self.prefix+'extreme_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
            sample[self.prefix+'extreme_points'] = None
        else:
            _points = helpers.extreme_points(_target, self.pert)
            # sample[self.prefix+'extreme_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)
            sample[self.prefix+'extreme_points'] = _points
        return sample

    def __str__(self):
        return 'ExtremePoints:(sigma=' + str(self.sigma) + \
                    ', pert='+str(self.pert)+', elem='+str(self.elem)+')'

class ToImage(object):
    """Return the given elements between 0 and custom_max."""
    def __init__(self, norm_elem=['image'], custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        for elem in self.norm_elem:
            tmp = sample[elem]
            sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage'