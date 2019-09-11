import os
import sys
import copy
import time
from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def crop_like(src, tgt):
  src_sz = np.array(src.shape)
  tgt_sz = np.array(tgt.shape)
  crop = (src_sz[2:4]-tgt_sz[2:4]) // 2
  if (crop > 0).any():
    return src[:, :, crop[0]:src_sz[2]-crop[0], crop[1]:src_sz[3]-crop[1], ...]
  else:
    return src


def apply_kernels(kernel, noisy_data):
  kh, kw = kernel.shape[2:]
  bs, ci, h, w = noisy_data.shape
  ksize = int(np.sqrt(kernel.shape[1]))

  # Crop kernel and input so their sizes match
  needed = kh + ksize - 1
  if needed > h:
    crop = (needed - h) // 2
    if crop > 0:
      kernel = kernel[:, :, crop:-crop, crop:-crop]
    kh, kw = kernel.shape[2:]
  else:
    crop = (h - needed) // 2
    if crop > 0:
      noisy_data = noisy_data[:, :, crop:-crop, crop:-crop]

  # -------------------------------------------------------------------------
  # Vectorize the kernel tiles
  kernel = kernel.permute(0, 2, 3, 1)
  kernel = kernel.contiguous().view(bs, 1, kh, kw, ksize*ksize)

  # Split the input buffer in tiles matching the kernels
  tiles = noisy_data.unfold(2, ksize, 1).unfold(3, ksize, 1)
  tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)
  # -------------------------------------------------------------------------

  weighted_sum = th.sum(kernel*tiles, dim=4)
  
  return weighted_sum



def get(params):
  params = copy.deepcopy(params)  # do not touch the original
  model_name = params.pop("model", None)
  if model_name is None:
    raise ValueError("model has not been specified!")
  return getattr(sys.modules[__name__], model_name)(**params)





class BayerNetwork(nn.Module):
  """Released version of the network, best quality.

  This model differs from the published description. It has a mask/filter split
  towards the end of the processing. Masks and filters are multiplied with each
  other. This is not key to performance and can be ignored when training new
  models from scratch.
  """
  def __init__(self, depth=15, width=64):
    super(BayerNetwork, self).__init__()

    self.depth = depth
    self.width = width

    layers = OrderedDict([
        ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance
      ])
    for i in range(depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 4
      if i == depth-1:
        n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)
    self.residual_predictor = nn.Conv2d(width, 12, 1)
    self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(6, width, 3)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    features = self.main_processor(mosaic)
    filters, masks = features[:, :self.width], features[:, self.width:]
    filtered = filters * masks
    residual = self.residual_predictor(filtered)
    upsampled = self.upsampler(residual)

    # crop original mosaic to match output size
    cropped = crop_like(mosaic, upsampled)

    # Concated input samples and residual for further filtering
    packed = th.cat([cropped, upsampled], 1)

    output = self.fullres_processor(packed)

    return output





class XtransNetwork(nn.Module):
  """Released version of the network.

  There is no downsampling here.

  """
  def __init__(self, depth=11, width=64):
    super(XtransNetwork, self).__init__()

    self.depth = depth
    self.width = width

    layers = OrderedDict([])
    for i in range(depth):
      n_in = width
      n_out = width
      if i == 0:
        n_in = 3
      # if i == depth-1:
      #   n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(3+width, width, 3)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    features = self.main_processor(mosaic)

    # crop original mosaic to match output size
    cropped = crop_like(mosaic, features)

    # Concated input samples and residual for further filtering
    packed = th.cat([cropped, features], 1)

    output = self.fullres_processor(packed)

    return output






class BayerNetworkNoise(nn.Module):
  """Released version of the network, best quality.

  This model differs from the published description. It has a mask/filter split
  towards the end of the processing. Masks and filters are multiplied with each
  other. This is not key to performance and can be ignored when training new
  models from scratch.
  """
  def __init__(self, depth=15, width=64):
    super(BayerNetworkNoise, self).__init__()

    self.depth = depth
    self.width = width

    
    self.pack_mosaic = nn.Sequential(OrderedDict([
        ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance
      ]))

   
    layerspre = OrderedDict([])
    layerspre["preconv{}".format(0+1)] = nn.Conv2d(5, 128, 3)
    layerspre["relu{}".format(0+1)] = nn.ReLU(inplace=True)
    self.preconv = nn.Sequential(layerspre)
    
    
    layers = OrderedDict([])
    for i in range(1,depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 4
      if i == depth-1:
        n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)
    self.residual_predictor = nn.Conv2d(width, 12, 1)
    self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(6, width, 3)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

  def forward(self, samples):
    # 1/4 resolution features
    mosaic = samples["mosaic"]
    noise_level = samples["noise_level"]
    
    packed = self.pack_mosaic(mosaic)
    
    sig = noise_level.expand(packed.shape[0],1, packed.shape[2], packed.shape[3])
    packed = th.cat([packed, sig], dim=1)
    
    pre  = self.preconv(packed)

    pre = pre[:,0:64,:,:] * pre[:,64:128,:,:]
    
    features = self.main_processor(pre)
    filters, masks = features[:, :self.width], features[:, self.width:]
    filtered = filters * masks
    residual = self.residual_predictor(filtered)
    upsampled = self.upsampler(residual)

    # crop original mosaic to match output size
    cropped = crop_like(mosaic, upsampled)

    # Concated input samples and residual for further filtering
    packed = th.cat([cropped, upsampled], 1)

    output = self.fullres_processor(packed)

    return output
