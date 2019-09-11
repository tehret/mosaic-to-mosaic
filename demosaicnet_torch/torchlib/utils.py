import logging
import torch as th
from torch.autograd import Variable
import numpy as np
import time
import csv

log = logging.getLogger(__name__)

import os
import re

def make_variable(d, cuda=True, fp16=False):
  ret = {}
  for k in d.keys():
    if cuda and hasattr(d[k], "cuda"):
      ret[k] = d[k].cuda()
      if fp16:
        ret[k] = ret[k].half()
    else:
      ret[k] = d[k]
  return ret

class CSVLogger(object):
  def __init__(self, directory, keys, filename="log.csv"):
    if directory.startswith('~'):
       directory = os.path.expanduser(directory)
    os.makedirs(directory, exist_ok=True)
    self.logfile = open(os.path.join(directory, filename), 'a')
    self.writer = csv.DictWriter(self.logfile, keys)
    # TODO: only write header once
    # self.writer.writeheader()

    # TODO: make sure filename is csv, add max entries and list

    # TODO: add periodic flush, cycles thru multiple log files as write fill up

  def __del__(self):
    # graceful closing of file
    if self.logfile:
      self.logfile.close()

  def log(self, **kwargs):
    """Write a log entry from a dictionary"""
    self.writer.writerow(kwargs)



class Checkpointer(object):
  @staticmethod
  def __get_sorted_checkpoints(directory):
    reg = re.compile(r".*\.pth\.tar")
    all_checkpoints = [f for f in os.listdir(directory) if
        reg.match(f)]
    mtimes = []
    for f in all_checkpoints:
      mtimes.append(os.path.getmtime(os.path.join(directory, f)))

    mf = sorted(zip(mtimes, all_checkpoints))
    chkpts = [m[1] for m in reversed(mf)]
    return chkpts

  @staticmethod
  def get_meta(directory):
    """Fetch metadata from a checkpoint directory to restore model params."""
    all_checkpoints = Checkpointer.__get_sorted_checkpoints(directory)
    for f in all_checkpoints:
      try:
        chkpt = th.load(os.path.join(directory, f))
        meta = chkpt["meta_params"]
        return meta
      except Exception as e:
        print("could not get meta from checkpoint {}, moving on.".format(f))
        print(e)
    raise ValueError("could not get meta from directoy {}".format(directory))

  def __init__(self, directory, model, optimizer, 
               max_save=5,
               interval=None,
               meta_params=None,
               filename='ckpt.pth.tar', verbose=False):
    """
    If interval > 0, checkpoints every "interval seconds".
    """

    if directory.startswith('~'):
        directory = os.path.expanduser(directory)

    # self.log.debug("Creating output directory {}".format(output))
    if not os.path.exists(directory):
      os.makedirs(directory)

    self.model = model
    self.optimizer = optimizer
    self.max_save = max_save
    self.directory = directory
    self.filename = filename
    self.verbose = verbose
    self.meta_params = meta_params
    self.interval = interval

    if self.interval is not None:
      self.last_checkpoint_time = time.time()

    all_checkpoints = Checkpointer.__get_sorted_checkpoints(self.directory)

    reg_epoch = re.compile(r"epoch.*\.pth\.tar")
    reg_periodic = re.compile(r"periodic.*\.pth\.tar")
    self.old_epoch_files = sorted([c for c in all_checkpoints if
                                   reg_epoch.match(c)])
    self.old_timed_files = sorted([c for c in all_checkpoints if
                                   reg_periodic.match(c)])

  def load_latest(self, ignore_optim=False):
    all_checkpoints = Checkpointer.__get_sorted_checkpoints(self.directory)

    if len(all_checkpoints) == 0:
      return None, 0, 0

    for f in all_checkpoints:
      try:
        step, e = self.load_checkpoint(os.path.join(self.directory, f),
                                 ignore_optim=ignore_optim)
        return f, step, e
      except Exception as e:
        print(e)
        print("could not load latest checkpoint {}, moving on.".format(f))
    return None, 0, 0

  def save_checkpoint(self, step, epoch, filename):
    if self.optimizer is not None:
      optimizer_state = self.optimizer.state_dict()
    else:
      optimizer_state = {}
    th.save({ 
        'step': step,
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer' : optimizer_state,
        'meta_params': self.meta_params,
        }, os.path.join(self.directory, filename))

  def save_best(self, step, epoch):
    self.save_checkpoint(step, epoch, 'best.pth.tar')

  def periodic_checkpoint(self, step, epoch):
    now = time.time()
    if self.interval is None or (
      now - self.last_checkpoint_time < self.interval):
      return False
    self.last_checkpoint_time = now

    filename = 'periodic_{}.pth.tar'.format(
      time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    self.save_checkpoint(step, epoch, filename)

    if self.max_save > 0:
      if len(self.old_timed_files) >= self.max_save:
        self.delete_checkpoint(self.old_timed_files[0])
        self.old_timed_files = self.old_timed_files[1:]
      self.old_timed_files.append(filename)

    return True

  def delete_checkpoint(self, filename):
    try:
      os.remove(os.path.join(self.directory, filename))
    except:
      print("exception in chekpoint deletion.")
      pass

  def load_checkpoint(self, filename, ignore_optim=False):
    chkpt = th.load(filename)
    self.model.load_state_dict(chkpt["state_dict"])
    if self.optimizer is not None and not ignore_optim and chkpt["optimizer"]:
      self.optimizer.load_state_dict(chkpt["optimizer"])
    return chkpt["step"], chkpt["epoch"]

  def on_epoch_end(self, step, epoch):
    filename = 'epoch_{:03d}.pth.tar'.format(epoch+1)
    if self.verbose > 0:
      print('\nEpoch %i: saving model to %s' % (epoch+1, file))
    self.save_checkpoint(step, epoch, filename)
    if self.max_save > 0:
      if len(self.old_epoch_files) >= self.max_save:
        self.delete_checkpoint(self.old_epoch_files[0])
        self.old_epoch_files = self.old_epoch_files[1:]
      self.old_epoch_files.append(filename)

  # Load init weights from a source checkpoint
  def override_params(self, filename, ignore=None):
    ov_chkpt = th.load(filename)
    tgt = self.model.state_dict()
    src = ov_chkpt["state_dict"]
    names = []
    if ignore is not None:
      ignore = re.compile(ignore)

    for name, param in src.items():
      if name in tgt and tgt[name].shape == param.shape:
        if ignore is not None and ignore.match(name):
          continue
        s = "{:10.10s}".format(name)
        s += " {:.2f} ({:.2f})".format(tgt[name].cpu().mean(), tgt[name].cpu().std())
        tgt[name].copy_(param)
        s += " -> {:.2f} ({:.2f})".format(param.cpu().mean(), param.cpu().std())
        names.append(s)
    return names


class ExponentialMovingAverage(object):
  """Keeps track of exponential moving averages, for each key."""
  def __init__(self, keys, alpha=0.999):
    self.first_update = {k: True for k in keys}
    self.alpha = alpha
    self.values = {k: 0 for k in keys}

  def __getitem__(self, key):
    return self.values[key]

  def update(self, key, value):
    if self.first_update[key]:
      self.values[key] = value
      self.first_update[key] = False
    else:
      self.values[key] = self.values[key]*self.alpha + value*(1.0-self.alpha)


class Averager(object):
  """Keeps track of running averages, for each key."""
  def __init__(self, keys):
    self.values = {k: 0.0 for k in keys}
    self.counts = {k: 0 for k in keys}

  def __getitem__(self, key):
    if self.counts[key] == 0:
      return 0.0
    return self.values[key] * 1.0/self.counts[key]

  def reset(self):
    for k in self.values.keys():
      self.values[k] = 0.0
      self.counts[k] = 0

  def update(self, key, value, count=1):
    self.values[key] += value*count
    self.counts[key] += count


class Timer(object):
  """A simple named timer context.

  Usage:
    with Timer("header_name"):
      do_sth()
  """
  def __init__(self, header=""):
    self.header = header
    self.time = 0

  def __enter__(self):
    self.time = time.time()

  def __exit__(self, tpye, value, traceback):
    elapsed = (time.time()-self.time)*1000
    print("{}, {:.1f}ms".format(self.header, elapsed))


def params2image(p):
  p = p.cpu().numpy()
  mu = p.mean()
  std = p.std()
  if std > 0:
    p = (p-mu) / (2*std)
  if len(p.shape) == 4:
    # conv
    p = np.pad(p, (
      (0, 0),
      (0, 0),
      (0, 1),
      (0, 1),
      ), 'constant')
    co, ci, kh, kw = p.shape

    p = np.reshape(np.transpose(p, [0, 2, 1, 3]), [co*kh, ci*kw])

  p = np.clip(0.5*(p + 1.0), 0, 1)
  return p
