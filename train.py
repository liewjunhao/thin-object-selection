""" Train TOS-Net. """
import os
os.environ['OMP_NUM_THREADS'] = "8"
from collections import OrderedDict
from datetime import datetime
import glob
import numpy as np
import argparse
import random
import json

# PyTorch includes
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate, sigmoid
import torch.backends.cudnn as cudnn

# Custom includes
import dataloaders.thinobject5k as thinobject5k
import dataloaders.custom_transforms as tr
import dataloaders.helpers as helpers
from layers.loss import binary_cross_entropy_loss, dice_loss, bootstrapped_cross_entopy_loss
import networks.tosnet as tosnet

# Default settings
MODEL_NAME = 'TOSNet'
RANDOM_SEED = 1234
# Network-specific arguments
NUM_INPUTS = 4                  # input channels
NUM_CLASSES = 1                 # number of classes
BACKBONE = 'resnet50'           # backbone architecture
LR_SIZE = 512                   # size of context stream
# Training-specific arguments
NUM_THIN_SAMPLES = 4            # number of samples consisting of thin parts
NUM_NON_THIN_SAMPLES = 1        # number of samples consisting of non-thin parts
MIN_SIZE = 512                  # minimum image size allowed
MAX_SIZE = 1980                 # maximum image size allowed
ROI_SIZE = 512                  # patch size for training
NUM_EPOCHS = 50                 # number of epochs for training
BATCH_SIZE = 1                  # batch size for training
SNAPSHOT = 10                   # store a model every 'snapshot'
LEARNING_RATE = 1e-3            # learning rate for training
WEIGHT_DECAY = 0.0005           # weight decay for training
MOMENTUM = 0.9                  # momentum for training
NUM_WORKERS = 6                 # number of workers to read daaset
RELAX_CROP = 50                 # enlarge bbox by 'relax_crop' pixels
ZERO_PAD_CROP = True            # insert zero padding when cropping
ADAPTIVE_RELAX = True           # compute 'relax_crop' adaptively?
DISPLAY = 20                    # print stats every 'display' iterations
CONTEXT_LOSS = {'bbce': 1}                      # losses for training context branch
MASK_LOSS = {'bootstrapped_ce': 1, 'dice': 1}   # losses for training mask prediction
EDGE_LOSS = {'bbce': 1, 'dice': 1}              # losses for training hr edge branch
DATASET = ['thinobject5k']      # dataset for training
LOSS_AVERAGE = 'size'           # how to average the loss
LR_SCHEDULE = 'poly'            # learning rate scheduler
BOOTSTRAPPED_RATIO = 1./16      # multiplier for determining #pixels in bootstrapping

def parse_args():
    parser = argparse.ArgumentParser(description='Training PatchNet')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--num_inputs', type=int, default=NUM_INPUTS)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--backbone', type=str, default=BACKBONE)
    parser.add_argument('--lr_size', type=int, default=LR_SIZE)
    parser.add_argument('--num_thin_samples', type=int, default=NUM_THIN_SAMPLES)
    parser.add_argument('--num_non_thin_samples', type=int, default=NUM_NON_THIN_SAMPLES)
    parser.add_argument('--min_size', type=int, default=MIN_SIZE)
    parser.add_argument('--max_size', type=int, default=MAX_SIZE)
    parser.add_argument('--roi_size', type=int, default=ROI_SIZE)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--snapshot', type=int, default=SNAPSHOT)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--relax_crop', type=int, default=RELAX_CROP)
    parser.add_argument('--zero_pad_crop', type=bool, default=ZERO_PAD_CROP)
    parser.add_argument('--adaptive_relax', type=bool, default=ADAPTIVE_RELAX)
    parser.add_argument('--display', type=int, default=DISPLAY)
    parser.add_argument('--context_loss', type=json.load, default=CONTEXT_LOSS)
    parser.add_argument('--mask_loss', type=json.load, default=MASK_LOSS)
    parser.add_argument('--edge_loss', type=json.load, default=EDGE_LOSS)
    parser.add_argument('--dataset', type=str, nargs='+', default=DATASET)
    parser.add_argument('--loss_average', type=str, default=LOSS_AVERAGE)
    parser.add_argument('--lr_schedule', type=str, default=LR_SCHEDULE)
    parser.add_argument('--bootstrapped_ratio', type=float, default=BOOTSTRAPPED_RATIO)
    args = parser.parse_args()
    return args

def _visualize_minibatch(sample, args):
    """Visualize a minibatch for debugging purpose."""
    import matplotlib.pyplot as plt
    inputs      = sample['concat'].to(device)
    inputs_lr   = sample['concat_lr'].to(device)
    grads       = sample['grad'].to(device)
    gts         = sample['crop_gt'].to(device)
    gts_lr      = sample['lr_gt'].to(device)
    gts_edge    = sample['crop_gt_edge'].to(device)
    # Thresholding
    gts = torch.ge(gts, 0.5).float()
    gts_lr = torch.ge(gts_lr, 0.5).float()
    gts_edge = torch.ge(gts_edge, 0.5).float()

    # Read rois and rearrange the inputs
    rois = sample['rois'].float().view(-1, 4)
    batch_ind = torch.arange(args.batch_size).float().unsqueeze(1).repeat(1, 
                    num_patch).view(-1, 1) # attach batch id
    rois = torch.cat((batch_ind, rois), dim=1).cuda()
    inputs = inputs.view(num_batch, 4, args.roi_size, args.roi_size)
    gts = gts.view(num_batch, 1, args.roi_size, args.roi_size)
    gts_edge = gts_edge.view(num_batch, 1, args.roi_size, args.roi_size)
    grads = grads.view(num_batch, 4, args.roi_size, args.roi_size)

    # Run roialign
    from torchvision.ops import RoIAlign
    roipool = RoIAlign((args.roi_size, args.roi_size), 1.0, -1).cuda()
    patches = roipool(inputs_lr, rois)

    # Convert to numpy
    inputs_lr = inputs_lr.cpu().numpy()
    gts_lr = gts_lr.cpu().numpy()
    inputs = inputs.cpu().numpy()
    gts = gts.cpu().numpy()
    gts_edge = gts_edge.cpu().numpy()
    patches = patches.cpu().numpy()
    grads = grads.cpu().numpy()

    for i in range(args.batch_size):
        # Visualize the low-resolution input
        image = inputs_lr[i, :3].transpose(1,2,0)
        gt = gts_lr[i, 0, :]
        clicks = inputs_lr[i, 3, :]
        clicks = np.float32(clicks == clicks.max())
        overlay = helpers.show_mask_and_clicks(image, gt, clicks)
        plt.figure(); plt.imshow(overlay.astype(np.uint8))
        plt.savefig('figure/train_lr_{}.jpg'.format(i))

        # Visualize the crop patches
        plt.figure()
        for j in range(num_patch):
            ind = i * args.batch_size + j
            image = inputs[ind, :3, :].transpose(1,2,0)
            gt = gts[ind, 0, :]
            clicks = inputs[ind, 3, :]
            if clicks.max() != 0:
                clicks = np.float32(clicks == clicks.max())
            overlay = helpers.show_mask_and_clicks(image, gt, clicks)
            patch = patches[ind, :3, :].transpose(1,2,0)
            edge = gts_edge[ind, 0, :]
            grad = grads[ind, 0, :]
            plt.subplot(4, num_patch, j+1)
            plt.imshow(overlay.astype(np.uint8))
            plt.subplot(4, num_patch, num_patch+j+1)
            plt.imshow(patch.astype(np.uint8))
            plt.subplot(4, num_patch, 2*num_patch+j+1)
            plt.imshow(edge.astype(np.uint8))
            plt.subplot(4, num_patch, 3*num_patch+j+1)
            plt.imshow(grad.astype(np.uint8))
        plt.savefig('images/train_patches_{}.jpg'.format(i))
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    args = parse_args()
    p = OrderedDict()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
        p[arg] = getattr(args, arg)

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    device = torch.device('cuda')
    # manual_seed_all is turned off due to the inconsistencies reported in 
    # https://discuss.pytorch.org/t/random-seed-initialization/7854/14
    torch.cuda.manual_seed_all(args.random_seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    # Create directories
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_root = os.path.join(save_dir_root, 'weights')
    runs = glob.glob(os.path.join(save_dir_root, 'run_*'))
    runs.sort(key=helpers.natural_keys)
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    # Network definition
    if args.backbone == 'resnet50':
        tosnet.lr_size = args.lr_size
        net = tosnet.tosnet_resnet50(n_inputs=args.num_inputs,
                                     n_classes=args.num_classes,
                                     os=16, pretrained='imagenet')
    else:
        raise NotImplementedError
    net.to(device)

    # Define the optimizer
    train_params = [{'params': tosnet.get_1x_lr_params(net), 'lr': args.lr},
                    {'params': tosnet.get_10x_lr_params(net), 'lr': args.lr * 10}]
    optimizer = optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    p['optimizer'] = str(optimizer)

    # Setup data transformations
    composed_transforms = [
        tr.RandomHorizontalFlip(),
        tr.CropFromMask(
            crop_elems=['image', 'gt', 'thin', 'void_pixels'],
            relax=args.relax_crop,
            zero_pad=args.zero_pad_crop,
            adaptive_relax=args.adaptive_relax,
            prefix=''),
        tr.Resize(
            resize_elems=['image', 'gt', 'thin', 'void_pixels'],
            min_size=args.min_size,
            max_size=args.max_size),
        tr.ComputeImageGradient(elem='image'),
        tr.ExtremePoints(sigma=10, pert=5, elem='gt'),
        tr.GaussianTransform(
            tr_elems=['extreme_points'],
            mask_elem='gt',
            sigma=10,
            tr_name='points'),
        tr.RandomCrop(
            num_thin=args.num_thin_samples,
            num_non_thin=args.num_non_thin_samples,
            crop_size=args.roi_size,
            prefix='crop_',
            thin_elem='thin',
            crop_elems=['image', 'gt', 'points', 'void_pixels', 'image_grad']),
        tr.MatchROIs(crop_elem='gt', resolution=args.lr_size),
        tr.FixedResizePoints(
            resolutions={
                'extreme_points': (args.lr_size, args.lr_size)},
            mask_elem='gt',
            prefix='lr_'),
        tr.FixedResize(
            resolutions={
                'image' : (args.lr_size, args.lr_size),
                'gt'    : (args.lr_size, args.lr_size),
                'void_pixels': (args.lr_size, args.lr_size)},
            prefix='lr_'),
        tr.GaussianTransform(
            tr_elems=['lr_extreme_points'],
            mask_elem='lr_gt',
            sigma=10,
            tr_name='lr_points'),
        tr.ToImage(
            norm_elem=['crop_points', 'crop_image_grad', 'lr_points']),
        tr.ConcatInputs(
            cat_elems=['lr_image', 'lr_points'],
            cat_name='concat_lr'),
        tr.ConcatInputs(
            cat_elems=['crop_image', 'crop_points'],
            cat_name='concat'),
        tr.ConcatInputs(
            cat_elems=['crop_image', 'crop_image_grad'],
            cat_name='grad'),
        tr.ExtractEdge(mask_elems=['crop_gt']),
        tr.RemoveElements(
            rm_elems=['points', 'image', 'gt', 'void_pixels', 'thin', 'image_grad']),
        tr.ToTensor(excludes=['rois'])]
    composed_transforms_tr = transforms.Compose(composed_transforms)

    # Setup dataset
    if len(args.dataset) == 1 and args.dataset[0] == 'thinobject5k':
        db_train = thinobject5k.ThinObject5K(split='train', 
                        transform=composed_transforms_tr, use_thin=True)
    else:
        raise NotImplementedError

    p['dataset_train'] = str(db_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, \
                            num_workers=args.num_workers, drop_last=True)
    helpers.generate_param_report(os.path.join(save_dir, args.model_name + '.txt'), p)

    # Train variables
    num_img_tr = len(trainloader)
    num_patch = args.num_thin_samples + args.num_non_thin_samples
    num_batch = num_patch * args.batch_size
    print('Training network')
    net.train()
    
    # Main training loop
    for epoch in range(args.num_epochs):
        for ii, sample in enumerate(trainloader):

            ### Uncomment to visualize ###
            # _visualize_minibatch(sample, args)

            # Read inputs and groundtruths
            inputs      = sample['concat'].to(device)
            inputs_lr   = sample['concat_lr'].to(device)
            grads       = sample['grad'].to(device)
            voids       = sample['crop_void_pixels'].to(device) # NEW
            voids_lr    = sample['lr_void_pixels'].to(device) # NEW
            gts         = sample['crop_gt'].to(device)
            gts_lr      = sample['lr_gt'].to(device)
            gts_edge    = sample['crop_gt_edge'].to(device)
            # Threshold
            gts = torch.ge(gts, 0.5).float() 
            gts_lr = torch.ge(gts_lr, 0.5).float()
            gts_edge = torch.ge(gts_edge, 0.5).float()
            
            # Read rois and rearrange the inputs
            rois = sample['rois'].float().view(-1, 4)
            batch_ind = torch.arange(args.batch_size).float().unsqueeze(1).repeat(1, 
                            num_patch).view(-1, 1) # attach batch id
            rois = torch.cat((batch_ind, rois), dim=1).to(device)
            inputs = inputs.view(num_batch, 4, args.roi_size, args.roi_size)
            grads = grads.view(num_batch, 4, args.roi_size, args.roi_size)
            gts = gts.view(num_batch, 1, args.roi_size, args.roi_size)
            gts_edge = gts_edge.view(num_batch, 1, args.roi_size, args.roi_size)
            voids = voids.view(num_batch, 1, args.roi_size, args.roi_size)

            # Forward pass
            outs = net.forward(inputs, grads, inputs_lr, rois)
            outs_lr, outs, edges = outs
            outs_lr = interpolate(outs_lr, gts_lr.size()[2:], mode='bilinear',
                            align_corners=True)
            
            # Compute loss
            loss_lr = 0.0
            if 'bce' in args.context_loss:
                loss_lr += args.context_loss['bce'] * binary_cross_entropy_loss(outs_lr, 
                            gts_lr, class_balance=False, reduction='mean', 
                            average=args.loss_average, void_pixels=voids_lr)
            if 'bbce' in args.context_loss:
                loss_lr += args.context_loss['bbce'] * binary_cross_entropy_loss(outs_lr, 
                            gts_lr, class_balance=True, reduction='mean', 
                            average=args.loss_average, void_pixels=voids_lr)
            if 'bootstrapped_ce' in args.context_loss:
                loss_lr += args.context_loss['bootstrapped_ce'] * bootstrapped_cross_entopy_loss(outs_lr,
                            gts_lr, void_pixels=voids_lr, ratio=args.bootstrapped_ratio)
            if 'dice' in args.context_loss:
                loss_lr += args.context_loss['dice'] * dice_loss(torch.sigmoid(outs_lr), 
                            gts_lr, void_pixels=voids_lr)

            loss_hr = 0.0
            if 'bce' in args.mask_loss:
                loss_hr += args.mask_loss['bce'] * binary_cross_entropy_loss(outs, 
                            gts, class_balance=False, reduction='mean', 
                            average=args.loss_average, void_pixels=voids)
            if 'bbce' in args.mask_loss:
                loss_hr += args.mask_loss['bbce'] * binary_cross_entropy_loss(outs, 
                            gts, class_balance=True, reduction='mean', 
                            average=args.loss_average, void_pixels=voids)
            if 'bootstrapped_ce' in args.mask_loss:
                loss_hr += args.mask_loss['bootstrapped_ce'] * bootstrapped_cross_entopy_loss(outs,
                            gts, void_pixels=voids, ratio=args.bootstrapped_ratio)
            if 'dice' in args.mask_loss:
                loss_hr += args.mask_loss['dice'] * dice_loss(torch.sigmoid(outs), 
                            gts, void_pixels=voids)
            
            loss_edge = 0.0
            if 'bce' in args.edge_loss:
                loss_edge += args.edge_loss['bce'] * binary_cross_entropy_loss(edges, 
                            gts_edge, class_balance=False, reduction='mean', 
                            average=args.loss_average, void_pixels=None)
            if 'bbce' in args.edge_loss:
                loss_edge += args.edge_loss['bbce'] * binary_cross_entropy_loss(edges, 
                            gts_edge, class_balance=True, reduction='mean', 
                            average=args.loss_average, void_pixels=None)
            if 'bootstrapped_ce' in args.edge_loss:
                loss_edge += args.edge_loss['bootstrapped_ce'] * bootstrapped_cross_entopy_loss(edges,
                            gts_edge, void_pixels=None, ratio=args.bootstrapped_ratio)            
            if 'dice' in args.edge_loss:
                loss_edge += args.edge_loss['dice'] * dice_loss(torch.sigmoid(edges), 
                            gts_edge, void_pixels=None)
            loss = loss_lr + loss_hr + loss_edge

            # Backprop and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            if args.lr_schedule == 'poly':
                lr = helpers.adjust_lr_poly(optimizer, args.lr, 
                        ii+num_img_tr*epoch, ii+num_img_tr*args.num_epochs)

            # Print stuff every args.display iterations
            if ii % args.display == 0:
                print('{}, Epoch: {}/{}, Iter: {}/{}, '
                      'Loss: {}'.format(datetime.now().strftime('%b%d_%H-%M-%S'),
                        epoch+1, args.num_epochs, ii+1, num_img_tr, loss.item()))

        # Save the model
        if (epoch % args.snapshot) == args.snapshot -1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', args.model_name + '_epoch-' + str(epoch) + '.pth'))
    print('Done training.') 