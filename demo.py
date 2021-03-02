import os
from collections import OrderedDict
import tkinter as tk
import torch
import networks.tosnet as tosnet
from ui.app import InteractiveDemo

def preprocess():
    """ Load networks with pre-trained weights. """
    # Read the configuration file
    with open(os.path.join('weights/tosnet_ours/config.txt'), 'r') as f:
        lines = f.read().splitlines()
    cfg = OrderedDict()
    for line in lines:
        line = line.split(':')
        if len(line) == 2:
            arg, val = line[0], line[1]
            cfg[arg] = val
    
    # Load the corresponding arguments
    cfg['num_inputs'] = int(cfg['num_inputs'])
    cfg['num_classes'] = int(cfg['num_classes'])
    cfg['lr_size'] = int(cfg['lr_size'])
    cfg['min_size'] = int(cfg['min_size'])
    cfg['max_size'] = int(cfg['max_size'])
    cfg['relax_crop'] = int(cfg['relax_crop'])
    cfg['zero_pad_crop'] = True if cfg['zero_pad_crop'] == 'True' else False
    cfg['adaptive_relax'] = True if cfg['adaptive_relax'] == 'True' else False

    # Load network
    device = torch.device('cuda')
    tosnet.lr_size = cfg['lr_size']
    net = tosnet.tosnet_resnet50(
                n_inputs=cfg['num_inputs'],
                n_classes=cfg['num_classes'],
                os=16, pretrained=None)
    weights = 'weights/tosnet_ours/models/TOSNet_epoch-49.pth'
    print('Loading from snapshot: {}'.format(weights))
    net.load_state_dict(torch.load(weights, map_location=lambda storage, loc:storage))
    net.to(device)
    net.eval()

    return net, device, cfg

if __name__ == '__main__':
    args = preprocess()
    app = InteractiveDemo()
    app._init_seg_model(*args)
    app.mainloop()