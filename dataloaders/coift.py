""" Dataset from COIFT.
Reference:
[1] Mansilla et al. "Object Segmentation by Oriented Image Foresting Transform 
    with Connectivity Transform", ICIP 2016.
[2] http://www.vision.ime.usp.br/~lucyacm/thesis/coift.html
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import pickle
import torch
import torch.utils.data as data
# Hack to solve the problem "cannot find mypath"
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from mypath import Path

class COIFT(data.Dataset):
    def __init__(
        self,
        split='test',
        root=Path.db_root_dir('coift'),
        transform=None,
        area_thres=0,
        retname=True):

        # Setup
        self.split = split
        self.root = root
        self.transform = transform
        self.area_thres = area_thres
        self.retname = retname
        self.imgs_dir = os.path.join(self.root, 'images')
        self.gts_dir = os.path.join(self.root, 'masks')

        if self.area_thres != 0:
            self.obj_list_file = os.path.join(self.root, self.split + \
                '_instances_area_thres-' + str(area_thres) + '.pkl')
        else:
            self.obj_list_file = os.path.join(self.root, self.split + \
                '_instances.pkl')

        # Read the list of dataset
        with open(os.path.join(self.root, 'list', split + '.txt')) as f:
            lines = f.read().splitlines()

        # Get the list of all images
        self.im_ids, self.imgs, self.gts = [], [], []
        for _line in lines:
            _img = os.path.join(self.imgs_dir, _line+'.jpg')
            _gt = os.path.join(self.gts_dir, _line+'.png')
            assert os.path.isfile(_img)
            assert os.path.isfile(_gt)
            self.im_ids.append(_line)
            self.imgs.append(_img)
            self.gts.append(_gt)
        assert (len(self.imgs) == len(self.gts))

        # Pre-compute the list of objects for each image
        if (not self._check_preprocess()):
            print('Preprocessing COIFT dataset, this will take long, but it '
                  'will be done only once.')
            self._preprocess()

        # Build the list of objects
        self.obj_list = []
        num_images = 0
        for ii in range(len(self.im_ids)):
            if self.im_ids[ii] in self.obj_dict.keys():
                for jj in self.obj_dict[self.im_ids[ii]].keys():
                    if self.obj_dict[self.im_ids[ii]][jj] != -1:
                        self.obj_list.append([ii, jj])
                num_images += 1

        # Display dataset statistics
        print('COIFT dataset\n'
              '-------------\n'
              '#Images: {:d}\n'
              '#Objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        _img, _gt = self._make_img_gt_pair(index)

        sample = {'image': _img,
                  'gt': _gt,
                  'void_pixels': np.zeros_like(_gt)}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'object': str(_obj_ii),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_preprocess(self):
        # Check that the file with categories is there and with correct size
        _obj_list_file = self.obj_list_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = np.load(self.obj_list_file, allow_pickle=True)
            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess(self):
        self.obj_dict = {}
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _gt = np.array(Image.open(self.gts[ii]))
            _gt_ids = np.unique(_gt)
            _gt_ids = _gt_ids[_gt_ids != 0]

            self.obj_dict[self.im_ids[ii]] = {}
            for jj in _gt_ids:
                obj_jj = np.float32(_gt == jj)
                if obj_jj.sum() > self.area_thres:
                    _obj_id = 1
                else:
                    obj_id = -1
                self.obj_dict[self.im_ids[ii]][jj] = _obj_id

        with open(self.obj_list_file, 'wb') as f:
            pickle.dump(self.obj_dict, f, pickle.HIGHEST_PROTOCOL)
        print('Preprocessing finished')

    def _make_img_gt_pair(self, index):
        _img_ii = self.obj_list[index][0]
        _obj_id = self.obj_list[index][1]

        # Read image and ground truth mask
        _img = np.float32(Image.open(self.imgs[_img_ii]))
        _gt = np.float32(Image.open(self.gts[_img_ii]))
        _gt = np.float32(_gt == _obj_id)
        return _img, _gt

    def __str__(self):
        return 'COIFT'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    import dataloaders.custom_transforms as tr
    from torchvision import transforms
    import dataloaders.helpers as helpers
    import imageio

    transform = transforms.Compose([tr.ToTensor()])
    dataset = COIFT(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for ii, sample in enumerate(dataloader):
        img = sample['image'].cpu().numpy()[0,:].transpose(1,2,0)
        gt = sample['gt'].cpu().numpy().squeeze()
        metas = sample['meta']

        overlay = helpers.mask_image(img.copy(), gt, color=[255,0,0], alpha=0.5)
        imageio.imwrite('images/sample.jpg', np.uint8(overlay))
        import pdb; pdb.set_trace()
