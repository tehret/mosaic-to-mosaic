from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from torchlib.image import crop_like


class MedianFilter(nn.Module):
  def __init__(self, ksize=3):
    super(MedianFilter, self).__init__()
    self.ksize = ksize

  def forward(self, x):
    k = self.ksize
    assert(len(x.shape) == 4)
    x = F.pad(x, [k//2, k//2, k//2, k//2])
    x = x.unfold(2, k, 1).unfold(3, k, 1)
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


class ImageGradients(nn.Module):
  def __init__(self, c_in):
    super(ImageGradients, self).__init__()
    self.dx = nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=False, groups=c_in)
    self.dy = nn.Conv2d(c_in, c_in, [3, 3], padding=1, bias=False, groups=c_in)


    self.dx.weight.requires_grad = False
    self.dy.weight.requires_grad = False

    self.dx.weight.data.zero_()
    self.dx.weight.data[:, :, 0, 0]  = -1
    self.dx.weight.data[:, :, 0, 2]  = 1
    self.dx.weight.data[:, :, 1, 0]  = -2
    self.dx.weight.data[:, :, 1, 2]  = 2
    self.dx.weight.data[:, :, 2, 0]  = -1
    self.dx.weight.data[:, :, 2, 2]  = 1

    self.dy.weight.data.zero_()
    self.dy.weight.data[:, :, 0, 0]  = -1
    self.dy.weight.data[:, :, 2, 0]  = 1
    self.dy.weight.data[:, :, 0, 1]  = -2
    self.dy.weight.data[:, :, 2, 1]  = 2
    self.dy.weight.data[:, :, 0, 2]  = -1
    self.dy.weight.data[:, :, 2, 2]  = 1

  def forward(self, im):
    return th.cat([self.dx(im), self.dy(im)], 1)

