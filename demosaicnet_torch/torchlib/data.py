class MultiDataLoader(object):
  class _MultiDataLoaderIter(object):
    def __init__(self, multi_loader):
      self.multi_loader = multi_loader
      self.iters = [iter(l) for l in self.multi_loader.loaders]
      self.current = 0

    def __iter__(self):
      return self

    def __next__(self):
      ret = self.iters[self.current].next()
      # Cycle through loaders
      self.current = (self.current + 1) % len(self.iters)
      return ret

  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return MultiDataLoader._MultiDataLoaderIter(self)

  def __len__(self):
    return sum([len(loader) for loader in self.loaders])
