# from collections import OrderedDict
#
# import numpy as np
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_scatter import scatter_add
#
# from torch.autograd import Variable
#
# from torchlib.image import crop_like
#
#
# class ApplyKernels(nn.Module):
#   """Gather values from tensor, weighted by kernels."""
#   def __init__(self, ksize, normalization=None):
#     super(ApplyKernels, self).__init__()
#     self.ksize = ksize
#
#     if normalization is None:
#       assert normalization in ["l1", "sum"], "unknown normalization {}, should be `l1` or `sum`".format(normalization)
#     self.normalization = normalization
#
#   def forward(self, kernel, tensor):
#     """
#     @param kernel:  Kernels to apply to tensor. spatial dimensions are linearized.
#     @type  kernel:  tensor, size [bs, k*k, h, w]
#
#     @param tensor:  tensor on which to apply the kernels
#     @type  tensor:  tensor, size [bs, c, h, w]
#     
#     @return:  a weighted reduction of `tensor` with weights in `kernels` and
#               the sum of weights (None is no normalization)
#     @rtype :  tensor
#     """
#     
#     kh, kw = kernel.shape[2:]
#     bs, ci, h, w = tensor.shape
#     ksize = self.ksize
#
#     # Crop kernel and input so their sizes match
#     needed = kh + ksize - 1
#     if needed > h:
#       crop = (needed - h) // 2
#       if crop > 0:
#         kernel = kernel[:, :, crop:-crop, crop:-crop]
#       kh, kw = kernel.shape[2:]
#     else:
#       crop = (h - needed) // 2
#       if crop > 0:
#         tensor = tensor[:, :, crop:-crop, crop:-crop]
#
#     # -------------------------------------------------------------------------
#     # Vectorize the kernel tiles
#     kernel = kernel.permute(0, 2, 3, 1)
#     kernel = kernel.contiguous().view(bs, 1, kh, kw, ksize*ksize)
#
#     # Split the input buffer in tiles matching the kernels
#     tiles = tensor.unfold(2, ksize, 1).unfold(3, ksize, 1)
#     tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)
#     # -------------------------------------------------------------------------
#
#     weighted_sum = th.sum(kernel*tiles, dim=4)
#     
#     if self.normalization == "sum":
#       kernel_sum = th.sum(kernel, dim=4)
#     elif self.normalization == "l1":
#       kernel_sum = th.sum(th.abs(kernel), dim=4)
#     elif self.normalization == "none":
#       kernel_sum = None
#
#     return weighted_sum, kernel_sum
#
#
# class ScatterSamples(nn.Module):
#   """Scatter values from samples, weighted by kernels."""
#
#   def __init__(self, ksize, crop=True):
#     super(ScatterSamples, self).__init__()
#     self.ksize = ksize
#     self.crop = crop
#
#   def forward(self, kernels, samples):
#     k = self.ksize
#     bs, _, h, w = kernels.shape
#
#     # Padding to account for kernel on the boundary
#     crop = k // 2
#     ph = h + 2*crop
#     pw = w + 2*crop
#
#     _, chans, rh, rw = samples.shape
#
#     samples = crop_like(samples, kernels)
#
#     kernels = kernels.view(bs, 1, k*k, h, w)
#     samples = samples.view(bs, chans, 1, h, w)
#
#     weighted = kernels*samples
#
#     # Coordinate of the sample w.r.t. in the padded output image
#     xx, yy = np.meshgrid(range(crop, ph-crop), range(crop, pw-crop))
#     idx = th.from_numpy(xx + pw*yy).view(1, 1, h, w).repeat(bs, k*k, 1, 1)
#
#     idx = idx.contiguous().view(bs, k*k, h*w)
#
#     # Coordinate within the kernel
#     kx, ky = np.meshgrid(range(-(k//2), k//2+1), range(-(k//2), k//2+1))
#
#     # Corresponding index shift in the padded output
#     shift = kx + ky*pw
#     shift = th.from_numpy(shift).view(1, k*k, 1)
#
#     # Absolute coordinate in the padded output, linearize the spatial index
#     neigh_idx = (idx + shift).view(bs, 1, k*k * h*w).repeat(1, chans, 1)
#
#     weighted = weighted.contiguous().view(bs, chans, k*k * h*w)
#
#     # Prepare the padded output
#     out = th.zeros(bs, chans, ph*pw).cuda()
#
#     neigh_idx = neigh_idx.cuda()
#
#     # Scatter samples
#     scatter_add(weighted, neigh_idx, out=out)
#
#     out = out.view(bs, chans, ph, pw)
#
#     if rh >= ph and rw >= pw:
#       # All good, we output as many or less pixels than were input
#       pass
#     elif rh < ph and rw < pw:
#       crop_h = (ph-rh) // 2 + k // 2
#       crop_w = (pw-rw) // 2 + k // 2
#       out = out[..., crop_h:crop_h+rh, crop_w:crop_w+rw]
#       # we need to crop the excess padding
#     else:
#       raise ValueError("not handled")
#
#     if self.crop:
#       out = out[..., k//2:-k//2-1, k//2:-k//2-1]
#
#     return out
