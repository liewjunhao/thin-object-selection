""" Thin Object Selection Network (TOS-Net). """

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from torchvision.ops import RoIAlign as ROIAlign
import os, sys
# HACK to solve the problem "cannot find layers"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import layers.layers_WS as L


model_urls = {'resnet50': 'weights/resnet50-19c8e357.pth'}
Conv2d = L.Conv2d
norm_layer = L.GroupNorm
lr_size = 416 # size of the low-resolution input

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    """ 3x3 convolution with padding. """
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=padding, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution. """
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, 1, dilation, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        
        out += residual
        out = self.relu(out)
        
        return out

class Deeplabv3PlusDecoderHead(nn.Module):
    """ Decoder head from DeepLabv3+. <https://arxiv.org/abs/1802.02611>
    Reference: 
    [1] Chen et al. "Encoder-Decoder with Atrous Separable Convolution 
    for Semantic Image Segmentation", ECCV 2018."""

    def __init__(self, dilations, num_classes):
        super(Deeplabv3PlusDecoderHead, self).__init__()

        self.assp = []
        self.assp = nn.ModuleList([self._make_assp(2048, 256, dilation) 
                        for dilation in dilations])
        self.image_pool = self._make_image_fea(2048, 256)
        self.encode_fea = self._make_block(256*5, 256, ks=1, pad=0)
        self.low_level_fea = self._make_block(256, 48, ks=1, pad=0)
        self.decode_1 = self._make_block(256+48, 256, ks=3, pad=1)
        self.decode_2 = self._make_block(256, 256, ks=3, pad=1)
        self.final = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)

    def _make_assp(self, in_features, out_features, dilation):
        if dilation == 1:
            conv = conv1x1(in_features, out_features)
        else:
            conv = conv3x3(in_features, out_features, 1, dilation, dilation)
        bn = norm_layer(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def _make_image_fea(self, in_features, out_features):
        g_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv = conv1x1(in_features, out_features)
        bn = norm_layer(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(g_pool, conv, bn, relu)

    def _make_block(self, in_features, out_features, ks=1, pad=0):
        conv = Conv2d(in_features, out_features, kernel_size=ks, padding=pad, 
                      stride=1, bias=False)
        bn = norm_layer(out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, res2_fea, res5_fea):
        # Get encoder feature
        h, w = res5_fea.size(2), res5_fea.size(3)
        image_fea = F.interpolate(input=self.image_pool(res5_fea), 
                        size=(h, w), mode='bilinear', align_corners=True)
        encoder_fea = [assp_stage(res5_fea) for assp_stage in self.assp]
        encoder_fea.append(image_fea)
        encoder_fea = self.encode_fea(torch.cat(encoder_fea, dim=1))

        # Get low level feature
        low_level_fea = self.low_level_fea(res2_fea)

        # Concat and decode
        h, w = res2_fea.size(2), res2_fea.size(3)
        high_level_fea = F.interpolate(input=encoder_fea, size=(h, w), 
                            mode='bilinear', align_corners=True)
        decoder_fea = self.decode_1(torch.cat((high_level_fea, low_level_fea), dim=1))
        decoder_fea = self.decode_2(decoder_fea)
        out = self.final(decoder_fea)
        return out, encoder_fea, decoder_fea

class EncoderBlock(nn.Module):
    def __init__(self, n_inputs, n_channels, n_side_channels, n_layers=2,
                 pool=True, scale=1.0):
        super(EncoderBlock, self).__init__()
        layers = [self._make_block(n_inputs, n_channels, ks=3)]
        for n in range(n_layers):
            layers.append(self._make_block(n_channels, n_channels, ks=3))
        self.main = nn.Sequential(*layers)
        self.side = self._make_block(n_side_channels, n_channels, ks=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2, padding=0) if pool else None
        self.scale = scale

    def _make_block(self, in_channels, out_channels, ks=1):
        if ks == 1:
            conv = conv1x1(in_channels, out_channels)
        elif ks == 3:
            conv = conv3x3(in_channels, out_channels)
        else:
            raise NotImplementedError
        norm = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, norm, relu)

    def forward(self, x, x_side, roi=None):
        x = self.main(x)
        xp = self.pool(x) if self.pool is not None else x
        
        x_side = self.side(x_side)
        if roi is None:
            roi = torch.Tensor([[0, 0, 0, lr_size, lr_size]]).to(x_side.device)
        h, w = xp.size()[2:]
        x_side = ROIAlign((h, w), self.scale, -1)(x_side, roi)

        xp = torch.cat((xp, x_side), dim=1)
        return xp, x

class DecoderBlock(nn.Module):
    def __init__(self, n_inputs, n_channels, n_layers=2):
        super(DecoderBlock, self).__init__()
        self.layer1 = self._make_block(n_inputs, n_channels, ks=1)
        layers = []
        for n in range(n_layers):
            layers.append(self._make_block(n_channels, n_channels, ks=3))
        self.layer2 = nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, ks=1):
        if ks == 1:
            conv = conv1x1(in_channels, out_channels)
        elif ks == 3:
            conv = conv3x3(in_channels, out_channels)
        else:
            raise NotImplementedError
        norm = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, norm, relu)

    def forward(self, x, x_side):
        x = self.layer1(x)
        x = F.interpolate(x, size=x_side.size()[2:], mode='bilinear', 
                align_corners=True)
        x = x + x_side # TODO: try concat instead of sum
        x = self.layer2(x)
        return x

class SimpleBottleneck(nn.Module):
    """Similar structure to the bottleneck layer of ResNet but with fixed #channels. """
    def __init__(self, planes):
        super(SimpleBottleneck, self).__init__()
        self.conv1 = conv1x1(planes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class TOSNet(nn.Module):
    def __init__(self, block, layers, n_inputs, n_classes, os=16, _print=True):
        super(TOSNet, self).__init__()
        self.inplanes = 64

        if _print:
            print('Constructing TOS-Net...')
            print('#Inputs: {}'.format(n_inputs))
            print('#Classes: {}'.format(n_inputs))
            print('Output Stride: {}'.format(os))

        # Setting dilated convolution rates
        if os == 16:
            layer3_stride, layer4_stride = 2, 1
            layer3_dilation = [1]*layers[2]
            layer4_dilation = [2, 4, 8]
        elif os == 8:
            layer3_stride, layer4_stride = 1, 1
            layer3_dilation = [2]*layers[2]
            layer4_dilation = [4, 8, 16]
        else:
            raise ValueError('Unsupported output stride: {}'.format(os))

        # Context branch (ResNet)
        self.conv1 = Conv2d(n_inputs, self.inplanes, kernel_size=7, stride=2, 
                            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, layers[0], stride=1, 
                                       dilation=[1]*layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       dilation=[1]*layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=layer3_stride, 
                                       dilation=layer3_dilation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer4_stride, 
                                       dilation=layer4_dilation)
        self.layer5 = Deeplabv3PlusDecoderHead((1,6,12,18), n_classes)

        # Edge branch
        n_layers = 2
        self.grad1 = EncoderBlock(4, 16, 256, n_layers, True, 1./4)
        self.grad2 = EncoderBlock(32, 32, 512, n_layers, True, 1./8)
        self.grad3 = EncoderBlock(64, 64, 1024, n_layers, True, 1./16)
        self.grad4 = EncoderBlock(128, 128, 2048, n_layers, True, 1./16)
        self.grad5 = EncoderBlock(256, 128, 256, n_layers, False, 1./16)
        self.grad4_decoder = DecoderBlock(256, 128, n_layers)
        self.grad3_decoder = DecoderBlock(128, 64, n_layers)
        self.grad2_decoder = DecoderBlock(64, 32, n_layers)
        self.grad1_decoder = DecoderBlock(32, 16, n_layers)
        self.edge = nn.Conv2d(16, n_classes, kernel_size=1, bias=True)

        # Fusion block
        self.mask_trans = nn.Sequential(conv1x1(256, 48),
                                        norm_layer(48),
                                        nn.ReLU(inplace=True))
        self.img_trans = nn.Sequential(conv1x1(3, 3),
                                       norm_layer(3),
                                       nn.ReLU(inplace=True))
        self.fuse0 = nn.Sequential(conv1x1(48+16+3, 16),
                                   norm_layer(16),
                                   nn.ReLU(inplace=True))
        self.fuse1 = SimpleBottleneck(16)
        self.fuse2 = SimpleBottleneck(16)
        self.fuse3 = SimpleBottleneck(16)
        self.mask = nn.Conv2d(16, n_classes, kernel_size=1, bias=True)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation[0], 
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation[i]))

        return nn.Sequential(*layers)

    def forward(self, x, x_grad, x_lr, roi=None):
        # Context stream
        x_lr0 = self.conv1(x_lr)
        x_lr0 = self.bn1(x_lr0)
        x_lr0 = self.relu(x_lr0)
        x_lr0 = self.maxpool(x_lr0)
        x_lr1 = self.layer1(x_lr0)
        x_lr2 = self.layer2(x_lr1)
        x_lr3 = self.layer3(x_lr2)
        x_lr4 = self.layer4(x_lr3)
        mask_lr, x_lr5_aspp, x_lr5 = self.layer5(x_lr1, x_lr4)

        # Edge stream
        x_gr1, x_enc1 = self.grad1(x_grad, x_lr1, roi)
        x_gr2, x_enc2 = self.grad2(x_gr1, x_lr2, roi)
        x_gr3, x_enc3 = self.grad3(x_gr2, x_lr3, roi)
        x_gr4, x_enc4 = self.grad4(x_gr3, x_lr4, roi)
        x_gr5, x_enc5 = self.grad5(x_gr4, x_lr5_aspp, roi)
        dec = self.grad4_decoder(x_gr5, x_enc4)
        dec = self.grad3_decoder(dec, x_enc3)
        dec = self.grad2_decoder(dec, x_enc2)
        dec = self.grad1_decoder(dec, x_enc1)
        edge = self.edge(dec)

        # Fusion stream
        x_lr5 = self.mask_trans(x_lr5)
        x_img = self.img_trans(x[:, :3, :])
        if roi is None:
            roi = torch.Tensor([[0, 0, 0, lr_size, lr_size]]).to(x.device)
        h, w = x.size()[2:]
        roipool = ROIAlign((h, w), 1./4, -1)
        x_lr5 = roipool(x_lr5, roi)
        x_img = roipool(x_img, roi)
        fuse0 = torch.cat((x_lr5, dec, x_img), dim=1)
        fuse0 = self.fuse0(fuse0)
        fuse1 = self.fuse1(fuse0)
        fuse2 = self.fuse2(fuse1)
        fuse3 = self.fuse3(fuse2)
        mask = self.mask(fuse3)

        return mask_lr, mask, edge

    def load_pretrained_weights(self, n_inputs, pretrained, backbone):
        if pretrained == 'imagenet':
            pth_model = model_urls[backbone]
        else:
            raise ValueError('Unknown pretrained weights: {}'.format(pretrained))
        print('Initializing from {}'.format(pretrained))
        saved_state_dict = torch.load(pth_model, map_location=lambda storage, loc:storage)

        new_layers = []
        params = deepcopy(self.state_dict())
        for i in saved_state_dict:
            if i.startswith('conv1') and n_inputs != 3:
                params[i][:, :3, :] = deepcopy(saved_state_dict[i])
            elif i in params.keys():
                params[i] = deepcopy(saved_state_dict[i])
            else:
                if 'num_batches_tracked' not in i:
                    new_layers.append(i)
        print('Pre-trained weights not used: {}'.format(new_layers))
        self.load_state_dict(params)

def get_1x_lr_params(model):
    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    b = [model.layer5,
         # Edge branch
         model.grad1, model.grad2, model.grad3, model.grad4, model.grad5,
         model.grad1_decoder, model.grad2_decoder, model.grad3_decoder,
         model.grad4_decoder, model.edge,
         # Fusion branch
         model.mask_trans, model.img_trans, 
         model.fuse0, model.fuse1, model.fuse2, model.fuse3, model.mask]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def tosnet_resnet50(n_inputs, n_classes, os=16, pretrained='imagenet'):
    model = TOSNet(Bottleneck, [3, 4, 23, 3], n_inputs=n_inputs, 
                   n_classes=n_classes, os=os)
    if pretrained is not None:
        model.load_pretrained_weights(n_inputs, pretrained, backbone='resnet50')
    return model


if __name__ == '__main__':
    x = torch.rand(1, 4, 512, 512).cuda()
    x_grad = torch.rand(1, 4, 512, 512).cuda()
    x_lr = torch.rand(1, 4, 416, 416).cuda()
    model = tosnet_resnet50(n_inputs=4, n_classes=1).cuda()
    model.eval()
    print(model)
    outputs = model(x, x_grad, x_lr, roi=None)
    for output in outputs:
        print(output.size())