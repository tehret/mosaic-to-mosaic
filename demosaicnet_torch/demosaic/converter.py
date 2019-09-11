import os
import numpy as np
import torch as th

class Converter(object):
  def __init__(self, pretrained_dir, model_type):
    self.basedir = pretrained_dir

  def convert(self, model):
    for n, p in model.named_parameters():
      name, tp = n.split(".")[-2:]

      old_name = self._remap(name)
      #print(old_name, "->", name)

      if tp == "bias":
        idx = 1
      else:
        idx = 0
      path = os.path.join(self.basedir, "{}_{}.npy".format(old_name, idx))
      data = np.load(path)
      # print(name, tp, data.shape, p.shape)

      # Overwiter
      #print(p.mean().item(), p.std().item())
      # import ipdb; ipdb.set_trace()
      #print(name, old_name, p.shape, data.shape)
      p.data.copy_(th.from_numpy(data))
      #print(p.mean().item(), p.std().item())

  def _remap(self, s):
    if s == "pack_mosaic":
      return "pack_mosaick"
    if s == "residual_predictor":
      return "residual"
    if s == "upsampler":
      return "unpack_mosaick"
    if s == "post_conv":
      return "post_conv1"
    return s
