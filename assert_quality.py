import os
import argparse
import numpy as np
import iio
from skimage.measure import compare_ssim

def psnr(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten() )
    return (10*np.log10(peak**2 / np.mean(x**2)))

def assert_quality(**args):
    imres = iio.read(args['input']).squeeze().astype(np.float)
    imref = iio.read(args['ref']).squeeze().astype(np.float)

    quant = psnr(imres, ref, 255.)
    quant_ssim = compare_ssim(imres, imref, data_range=255., multichannel=True)
    print(quant, quant_ssim)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Assert quality (PSNR + SSIM)")
    parser.add_argument("--input", type=str,
                        help='path to input image')
    parser.add_argument("--ref", type=str,
                        help='path to the reference image')
    argspar = parser.parse_args()

    print("\n### Assert quality (PSNR + SSIM) ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
            print('\t{}: {}'.format(p, v))
    print('\n')

    assert_quality(**vars(argspar))
