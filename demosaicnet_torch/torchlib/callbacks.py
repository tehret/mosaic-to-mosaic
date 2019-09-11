class Callback(object):
  def __init__(self):
    self.current_epoch = -1

  def on_epoch_begin(self, epoch):
    self.current_epoch = epoch

  def on_epoch_end(self, epoch, logs, batch_data=None):
    pass

  def on_batch_begin(self, batch, logs):
    pass

  def on_batch_end(self, batch, batch_id, num_batches, logs):
    pass

  def get_frac(self, batch, num_batches):
    frac = self.current_epoch + batch*1.0/num_batches
    return frac


class LossCallback(Callback):
  def __init__(self, env=None):
    super(LossCallback, self).__init__()

  def on_batch_end(self, batch, batch_id, num_batches, logs):
    frac = self.get_frac(batch_id, num_batches)

  def on_epoch_end(self, epoch, logs):
      return


class AccuracyCallback(Callback):
  def __init__(self, env=None):
    super(AccuracyCallback, self).__init__()

  def on_batch_end(self, batch, batch_id, num_batches, logs):
    frac = self.get_frac(batch_id, num_batches)

  def on_epoch_end(self, epoch, logs):
      return
