#!/usr/bin/env python
# MIT License
#
# Deep Joint Demosaicking and Denoising
# Siggraph Asia 2016
# Michael Gharbi, Gaurav Chaurasia, Sylvain Paris, Fredo Durand
# 
# Copyright (c) 2016 Michael Gharbi
# Modified Version: Copyright (c) 2019 Thibaud Ehret
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#!/usr/bin/env python
""""""

import argparse
import logging
import os
import setproctitle
import time

import numpy as np
import torch as th
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from torchlib.trainer import Trainer

import demosaic.dataset as dset
import demosaic.modules as modules
import demosaic.losses as losses
import demosaic.callbacks as callbacks
import demosaic.converter as converter
import torchlib.callbacks as default_callbacks

log = logging.getLogger("demosaick")

def main(args, model_params):
  if args.fix_seed:
    np.random.seed(0)
    th.manual_seed(0)

  # ------------ Set up datasets ----------------------------------------------
  xforms = [dset.ToTensor()]
  if args.green_only:
    xforms.append(dset.GreenOnly())
  xforms = transforms.Compose(xforms)
  if args.xtrans:
    data = dset.XtransDataset(args.data_dir, transform=xforms, augment=True, linearize=args.linear)
  else:
    data = dset.BayerDataset(args.data_dir, transform=xforms, augment=True, linearize=args.linear)
  data[0]

  if args.val_data is not None:
    if args.xtrans:
      val_data = dset.XtransDataset(args.val_data, transform=xforms, augment=False)
    else:
      val_data = dset.BayerDataset(args.val_data, transform=xforms, augment=False)
  else:
    val_data = None
  # ---------------------------------------------------------------------------

  model = modules.get(model_params)
  log.info("Model configuration: {}".format(model_params))

  if args.pretrained:
    log.info("Loading Caffe weights")
    if args.xtrans:
      model_ref = modules.get({"model": "XtransNetwork"})
      cvt = converter.Converter(args.pretrained, "XtransNetwork")
    else:
      model_ref = modules.get({"model": "BayerNetwork"})
      cvt = converter.Converter(args.pretrained, "BayerNetwork")
    cvt.convert(model_ref)
    model_ref.cuda()
  else:
    model_ref = None

  if args.green_only:
    model = modules.GreenOnly(model)
    model_ref = modules.GreenOnly(model_ref)

  if args.subsample:
    dx = 1
    dy = 0
    if args.xtrans:
      period = 6
    else:
      period = 2
    model = modules.Subsample(model, period, dx=dx, dy=dy)
    model_ref = modules.Subsample(model_ref, period, dx=dx, dy=dy)

  if args.linear:
    model = modules.DeLinearize(model)
    model_ref = modules.DeLinearize(model_ref)

  name = os.path.basename(args.output)
  cbacks = [
      default_callbacks.LossCallback(env=name),
      callbacks.PSNRCallback(env=name),
      ]

  metrics = {
      "psnr": losses.PSNR(crop=4)
      }

  log.info("Using {} loss".format(args.loss))
  if args.loss == "l2":
    criteria = { "l2": losses.L2Loss(), }
  elif args.loss == "l1":
    criteria = { "l1": losses.L1Loss(), }
  elif args.loss == "gradient":
    criteria = { 
      "gradient": losses.GradientLoss(), 
    }
  elif args.loss == "laplacian":
    criteria = { 
      "laplacian": losses.LaplacianLoss(), 
    }
  elif args.loss == "vgg":
    criteria = { "vgg": losses.VGGLoss(), }
  else:
    raise ValueError("not implemented")

  optimizer = optim.Adam
  optimizer_params = {}
  if args.optimizer == "sgd":
    optimizer = optim.SGD
    optimizer_params = {"momentum": 0.9}

  train_params = Trainer.Parameters(
      viz_step=100, lr=args.lr, batch_size=args.batch_size,
      optimizer=optimizer, optimizer_params=optimizer_params)

  trainer = Trainer(
      data, model, criteria, output=args.output, 
      params = train_params,
      model_params=model_params, verbose=args.debug, 
      callbacks=cbacks,
      metrics=metrics,
      valset=val_data, cuda=True)

  trainer.train()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # I/O params
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--val_data')
  parser.add_argument('--checkpoint')
  parser.add_argument('--pretrained')

  # Training
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--fix_seed', dest="fix_seed", action="store_true")
  parser.add_argument('--loss', default="l2", choices=["l1", "vgg", "l2", "gradient", "laplacian"])
  parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd"])

  # Monitoring
  parser.add_argument('--debug', dest="debug", action="store_true")

  # Model
  parser.add_argument('--xtrans', dest="xtrans", action="store_true")
  parser.add_argument('--green_only', dest="green_only", action="store_true")
  parser.add_argument('--subsample', dest="subsample", action="store_true")
  parser.add_argument('--linear', dest="linear", action="store_true")
  parser.add_argument(
      '--params', nargs="*", default=["model=BayerNetwork"])

  parser.set_defaults(debug=False, fix_seed=False, xtrans=False,
                      green_only=False, subsample=False, linear=False)

  args = parser.parse_args()

  params = {}
  if args.params is not None:
    for p in args.params:
      k, v = p.split("=")
      if v.isdigit():
        v = int(v)
      elif v == "False":
        v = False
      elif v == "True":
        v = True

      params[k] = v

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  setproctitle.setproctitle(
      'demosaic_{}'.format(os.path.basename(args.output)))

  main(args, params)
