import numpy as np
import torch as th

import torchlib.callbacks as callbacks
from torch.utils.data import DataLoader
import torchlib.utils as utils
from torchlib.image import crop_like
import demosaic.losses as losses

from torchlib.modules.image_processing import ImageGradients

class DemosaicVizCallback(callbacks.Callback):
  def __init__(self, data, model, ref, env=None, batch_size=16, 
               shuffle=False, cuda=True, period=500):
    super(DemosaicVizCallback, self).__init__()
    self.batch_size = batch_size
    self.model = model
    self.ref = ref
    self._cuda = cuda

    self.loader = DataLoader(
        data, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, drop_last=True)

    self.period = period
    self.counter = 0

    self.psnr = losses.PSNR(crop=8)

    self.grads = ImageGradients(3).cuda()

  # def on_epoch_end(self, epoch, logs):
  def on_batch_end(self, batch, batch_id, num_batches, logs):
    if self.counter % self.period != 0:
      self.counter += 1
      return

    self.counter = 0
    for (batch_id, batch) in enumerate(self.loader):
      # Get a batch
      batch_v = utils.make_variable(batch, cuda=self._cuda)

      # Forward
      output = self.model(batch_v)
      if self.ref is None:
        output_ref = th.zeros_like(output)
      else:
        output_ref = self.ref(batch_v)
        # make sure size match
        output_ref = crop_like(output_ref, output)
        output = crop_like(output, output_ref)


      eps = 1e-8
      mosaic = batch_v["mosaic"].data
      target = batch_v["target"].data
      # noise_variance = batch_v["noise_variance"].data
      output = output.data
      output_ref = output_ref.data
      target = crop_like(target, output)
      mosaic = crop_like(mosaic, output)

      diff = (output-target)

      gdiff = self.grads(diff).abs()
      gdiff = th.nn.functional.pad(gdiff, (0, 1, 0, 1))

      diff = diff.abs()

      gdiff_x = gdiff[:, :3]
      gdiff_y = gdiff[:, 3:]

      gdiff_x = gdiff_x / (target + 1e-4)
      gdiff_y = gdiff_y / (target + 1e-4)

      psnr = self.psnr(batch_v, output).item()
      psnr_ref = self.psnr(batch_v, output_ref).item()

      return  # process only one batch


class PSNRCallback(callbacks.Callback):
  def __init__(self, env=None):
    super(PSNRCallback, self).__init__()

  def on_batch_end(self, batch, batch_id, num_batches, logs):
    frac = self.get_frac(batch_id, num_batches)

  def on_epoch_end(self, epoch, logs):
    return

