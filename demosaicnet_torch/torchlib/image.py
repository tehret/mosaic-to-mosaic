import numpy as np

def crop_like(src, tgt):
  src_sz = np.array(src.shape)
  tgt_sz = np.array(tgt.shape)
  crop = (src_sz[-2:]-tgt_sz[-2:]) // 2
  if (crop > 0).any():
    return src[..., crop[0]:src_sz[-2]-crop[0], crop[1]:src_sz[-1]-crop[1]]
    # return src[:, :, crop[0]:src_sz[2]-crop[0], crop[1]:src_sz[3]-crop[1], ...]
  else:
    return src

def read_pfm(path):
  with open(path, 'rb') as fid:
    identifier = fid.readline().strip()
    if identifier == b'PF':  # color
      nchans = 3
    elif identifier == b'Pf':  # gray
      nchans = 1
    else:
      raise ValueError("Unknown PFM identifier {}".format(identifier))

    dimensions = fid.readline().strip()
    width, height = [int(x) for x in dimensions.split()]
    endianness = fid.readline().strip()

    data = np.fromfile(fid, dtype=np.float32, count=width*height*nchans)
    data = np.reshape(data, (height, width, nchans))

    data[np.isnan(data)] = 0.0

    return data

def read_ppm(path):
  with open(path, 'rb') as fid:
    identifier = fid.readline().strip()
    if identifier == b'P6':  # color
      nchans = 3
    # elif identifier == b'Pf':  # gray
    #   nchans = 1
    else:
      raise ValueError("Unknown PFM identifier {}".format(identifier))

    dimensions = fid.readline().strip().split()
    if len(dimensions) != 2:
      dimensions += fid.readline().strip().split()
    width, height = [int(x) for x in dimensions]
    maxval = int(fid.readline().strip())

    data = np.fromfile(fid, dtype=np.uint16, count=width*height*nchans)
    data.byteswap(inplace=True)
    data = np.reshape(data, (height, width, nchans))

    data = data.astype(np.float32) / (1.0*maxval)

    data[np.isnan(data)] = 0.0

    # import ipdb; ipdb.set_trace()

    return data

def write_pfm(path, im):
  assert im.dtype == np.float32, "pfm image should be float32"
  height, width, nchans = im.shape
  with open(path, 'wb') as fid:
    if nchans == 1:
      fid.write(b'Pf\n')
    elif nchans == 3:
      fid.write(b'PF\n')
    else:
      raise ValueError("Unknown channel count {}".format(nchans))

    # size
    s = str(width).encode() + b' ' + str(height).encode() + b'\n'
    fid.write(s)

    # endian
    fid.write(b'-1\n')

    im.tofile(fid)
