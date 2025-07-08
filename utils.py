import imp
import torch
from torch.nn import L1Loss, MSELoss
import nibabel as nib
import numpy as np
import cv2


class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        self.instance = L1Loss()

    def forward(self, target, output):
        l = self.instance(output[:,0,:,:], target[:,0,:,:])
        return l


def save_input(inputs, path):
    for i in range(inputs.shape[1]):
        tmp = inputs[0,i,:,:].detach().cpu().numpy()
        tmp = ( (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-10) * (2**16 - 1) ).astype('uint16')
        cv2.imwrite(path + '_{}.png'.format(i), tmp)

    return
