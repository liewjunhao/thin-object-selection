from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_cross_entropy_loss(output, label, void_pixels=None,
        class_balance=False, reduction='mean', average='batch'):
    """Binary cross entropy loss for training
    # arguments:
        output: output from the network
        label: ground truth label
        void_pixels: pixels to ignore when computing the loss
        class_balance: to use class-balancing weights
        reduction (str): either 'none'|'sum'|'mean'
            'none': the loss remains the same size as output
            'sum': the loss is summed
            'mean': the loss is average based on the 'average' flag
        average (str): either 'size'|'batch'
            'size': loss divide by #pixels
            'batch': loss divide by batch size
    Remarks: Currently, class_balance=True does not support
    reduction='none'
    """
    assert output.size() == label.size()
    assert not (class_balance and reduction == 'none')

    labels = torch.ge(label, 0.5).float()
    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    
    if class_balance:
        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

        if reduction == 'sum':
            # sum the loss across all elements
            return final_loss
        elif reduction == 'mean':
            # average the loss
            if average == 'size':
                final_loss /= num_total
            elif average == 'batch':
                final_loss /= label.size()[0]
            return final_loss
        else:
            raise ValueError('Unsupported reduction mode: {}'.format(reduction))
    
    else:
        loss_val = -loss_val
        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_val = torch.mul(w_void, loss_val)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
        # final_loss = torch.sum(loss_val)

        if reduction == 'none':
            # return the loss directly
            return loss_val
        elif reduction == 'sum':
            # sum the loss across all elements
            return torch.sum(loss_val)
        elif reduction == 'mean':
            # average the loss
            final_loss = torch.sum(loss_val)
            if average == 'size':
                final_loss /= num_total
            elif average == 'batch':
                final_loss /= label.size()[0]
            return final_loss
        else:
            raise ValueError('Unsupported reduction mode: {}'.format(reduction))


def dice_loss(output, label, void_pixels=None, smooth=1e-8):
    """Dice loss for training.
    Remarks:
    + Sigmoid should be applied before applying this loss.
    + This loss currently only supports average='size'.
    """
    assert output.size() == label.size()
    p2 = (output * output)
    g2 = (label * label)
    pg = (output * label)
    batch_size = output.size(0)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        p2 = torch.mul(p2, w_void)
        g2 = torch.mul(g2, w_void)
        pg = torch.mul(pg, w_void)

    p2 = p2.sum(3).sum(2).sum(1) # (N, )
    g2 = g2.sum(3).sum(2).sum(1) # (N, )
    pg = pg.sum(3).sum(2).sum(1) # (N, )
    final_loss = 1.0 - torch.div((2 * pg), (p2 + g2 + smooth))
    final_loss = torch.sum(final_loss)
    final_loss /= batch_size
    return final_loss

def three_class_balanced_softmax_ce_loss(output, label, reduction='mean',
                                         average='batch'):
    """Softmax cross-entropy loss with balancing weights.
    Note that this loss is designed for segmentation of 3 classes only.
    TODO: extend this to multiclass version.
    """
    assert output.size()[2:] == label.size()[2:]
    assert output.size(1) == 3

    num_classes = output.size(1)
    num_labels = [(label==ind).sum() for ind in range(3)]

    # compute the balancing weight for each class
    if num_labels[-1] == 0:
        w1 = num_labels[1]
        w2 = num_labels[0]
        ws = w1 + w2
        weight = torch.Tensor([w1/ws, w2/ws, 0]).to(output.device)
    else:
        w1 = num_labels[1] * num_labels[2]
        w2 = num_labels[0] * num_labels[2]
        w3 = num_labels[0] * num_labels[1]
        ws = (w1 + w2 + w3).float()
        weight = torch.Tensor([w1/ws, w2/ws, w3/ws]).to(output.device)

    criterion = nn.CrossEntropyLoss(weight=weight, reduction='sum')
    loss = criterion(output, label[:, 0, :])
    
    if reduction == 'sum':
        return loss
    elif reduction == 'mean':
        if average == 'size':
            num_labels = num_labels[0] + num_labels[1] + num_labels[2]
            loss /= torch.sum(num_labels)
        elif average == 'batch':
            loss /= label.size(0)
        return loss
    else:
        raise NotImplementedError

def bootstrapped_cross_entopy_loss(output, label, ratio=1./16, void_pixels=None):
    """Bootstrapped cross-entropy loss used in FRRN 
    <https://arxiv.org/abs/1611.08323>
    Reference:
        [1] Tobias et al. "Full-Resolution Residual Networks for Semantic 
        Segmentation in Street Scenes", CVPR 2017.
        [2] https://github.com/TobyPDE/FRRN/blob/master/dltools/losses.py
    Args:
        output: The output of the network
        label: The ground truth label
        batch_size: The batch size
        ratio: A variable that determines the number of pixels
               selected in the bootstrapping process. The number of pixels
               is determined by size**2 * ratio, where we assume the 
               height and width are the same (size).
    """
    # compute cross entropy
    assert output.size() == label.size()

    labels = torch.ge(label, 0.5).float()
    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    # compute the pixel-wise cross entropy.
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    
    # ignore the void pixels
    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_val = torch.mul(w_void, loss_val)
    
    xentropy = -loss_val

    # for each element in the batch, collect the top K worst predictions
    K = int(label.size(2) * label.size(3) * ratio)
    batch_size = label.size(0)
    
    result = 0.0
    for i in range(batch_size):
        batch_errors = xentropy[i, :]
        flat_errors = torch.flatten(batch_errors)

        # get the worst predictions.
        worst_errors, _ = torch.sort(flat_errors)[-K:]

        result += torch.mean(worst_errors)

    result /= batch_size

    return result