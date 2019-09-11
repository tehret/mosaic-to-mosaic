# coding=utf-8

#TODO

import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import skimage
import skimage.io
import tifffile
from scipy.ndimage.morphology import binary_dilation
import time
import piio
from skimage.measure import compare_ssim

import demosaic.modules as modules
import demosaic.converter as converter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_net(net_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    pretrained_base = here+'/pretrained_models/'

    print("Loading Caffe weights")
    model_name = "BayerNetworkNoise"
    model_path = pretrained_base+'/bayer_noise/'

    # if a network path is provided use it instead 
    if net_path: model_path = net_path

    model_ref = modules.get({"model": model_name})
    cvt = converter.Converter(model_path, model_name)
    cvt.convert(model_ref)

    return model_ref

def psnr(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten())
    return (10*np.log10(peak**2 / np.mean(x**2)))

def rgb2bayer(img):
    mask = torch.zeros(img.size()).cuda()

    # Red
    mask[0,0, 0::2, 1::2] = 1

    # green
    mask[0,1, 0::2, 0::2] = 1
    mask[0,1, 1::2, 1::2] = 1

    # blue
    mask[0,2, 1::2, 0::2] = 1

    return img*mask, mask

class WarpedLoss(nn.Module):
        def __init__(self):
            super(WarpedLoss, self).__init__()
            self.criterion = nn.L1Loss(reduction='sum')

        def cubic_interpolation(self,A,B,C,D,x):
            a,b,c,d = A.size()
            x = x.view(a,1,c,d).repeat(1,3,1,1)
            return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

        # Return the img interpolated on the given grid, return also a mask
        def bicubic_interpolation(self, im, grid):
            B, C, H, W = im.size()

            x0 = torch.floor(grid[0,0,:,:] - 1).long()
            y0 = torch.floor(grid[0,1,:,:] - 1).long()
            x1 = x0 + 1 
            y1 = y0 + 1 
            x2 = x0 + 2 
            y2 = y0 + 2 
            x3 = x0 + 3 
            y3 = y0 + 3 

            outsideX = torch.max(torch.gt(-x0, 0), torch.gt(x3, W-1)).repeat((B,C,1,1))
            outsideY = torch.max(torch.gt(-y0, 0), torch.gt(y3, H-1)).repeat((B,C,1,1))

            mask = torch.Tensor(np.ones((B, C, H, W))).cuda()
            mask[outsideX] = 0
            mask[outsideY] = 0

            x0 = x0.clamp(0,W-1)
            y0 = y0.clamp(0,H-1)
            x1 = x1.clamp(0,W-1)
            y1 = y1.clamp(0,H-1)
            x2 = x2.clamp(0,W-1)
            y2 = y2.clamp(0,H-1)
            x3 = x3.clamp(0,W-1)
            y3 = y3.clamp(0,H-1)

            A = self.cubic_interpolation(im[:,:,y0,x0],im[:,:,y1,x0],im[:,:,y2,x0],im[:,:,y3,x0], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
            B = self.cubic_interpolation(im[:,:,y0,x1],im[:,:,y1,x1],im[:,:,y2,x1],im[:,:,y3,x1], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
            C = self.cubic_interpolation(im[:,:,y0,x2],im[:,:,y1,x2],im[:,:,y2,x2],im[:,:,y3,x2], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
            D = self.cubic_interpolation(im[:,:,y0,x3],im[:,:,y1,x3],im[:,:,y2,x3],im[:,:,y3,x3], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
            
            output = self.cubic_interpolation(A,B,C,D,grid[:,0,:,:] - torch.floor(grid[:,0,:,:]))
            return output, mask

        # Warp the image according to the transform p (p can be translation up to homography).
        # It also returns a mask of unavailable pixels
        def warp(self, x, p, scale=1):
            B, C, H, W = x.size()
            # mesh grid 
            xx = torch.arange(0, scale*W).view(1,-1).repeat(np.round(scale*H).astype(np.int),1)
            yy = torch.arange(0, scale*H).view(-1,1).repeat(1,np.round(scale*W).astype(np.int))
            xx = xx.view(1,1,np.round(scale*H).astype(np.int),np.round(scale*W).astype(np.int)).repeat(B,1,1,1).float()
            yy = yy.view(1,1,np.round(scale*H).astype(np.int),np.round(scale*W).astype(np.int)).repeat(B,1,1,1).float()

            scale = 2
            # Rescale the grid to match the input image
            xx /= scale
            yy /= scale

            # Translation
            if len(p) == 2:
                xx_ = xx + p[0]
                yy_ = yy + p[1]
            # Euclidean transform
            elif len(p) == 3:
                xx_ = np.cos(p[2])*xx - np.sin(p[2])*yy + p[0]
                yy_ = np.sin(p[2])*xx + np.cos(p[2])*yy + p[1]
            # Similarity
            elif len(p) == 4:
                xx_ = (1 + p[2])*xx + p[3]*yy + p[0]
                yy_ = p[3]*xx + (1 + p[2])*yy + p[1]
            # Affinity
            elif len(p) == 6:
                xx_ = (1 + p[2])*xx + p[3]*yy + p[0]
                yy_ = p[4]*xx + (1 + p[5])*yy + p[1]
            # Homography
            elif len(p) == 8:
                d = p[6]*xx + p[7]*yy + 1
                xx_ = ((1 + p[0])*xx + p[1]*yy + p[2])/d
                yy_ = (p[3]*xx + (1 + p[4])*yy + p[5])/d
            # Homography (orsa version)
            elif len(p) == 9:
                d = p[6]*xx + p[7]*yy + 1
                xx_ = (p[0]*xx + p[1]*yy + p[2])/d
                yy_ = (p[3]*xx + p[4]*yy + p[5])/d

            xx_ *= scale
            yy_ *= scale

            grid = torch.cat((xx_,yy_),1)
            vgrid = Variable(grid.cuda())
            output, mask = self.bicubic_interpolation(x, vgrid)
            return output, mask

        def forward(self, input, target, p):
            # Warp input on target
            warped, mask = self.warp(input, p) 

            # Compute the bayer mask
            bayer_mask = torch.Tensor(np.zeros(input.size())).cuda()
            # Red
            bayer_mask[0,0, 0::2, 1::2] = 1
            # green
            bayer_mask[0,1, 0::2, 0::2] = 1
            bayer_mask[0,1, 1::2, 1::2] = 1
            # blue
            bayer_mask[0,2, 1::2, 0::2] = 1

            mask = bayer_mask * mask

            # Check if the target is a Bayer pattern (single channel
            # or a RGB image (three channels) used as a Bayer.
            # If it's a Bayer repeat it so it looks like RGB so we compare
            # tensors of same size
            _, C, _, _ = target.size()
            if C == 1:
                target = target.view(1, 3, 1, 1)
            # Nothing to do otherwise

            # Compute the masked loss
            self.loss = self.criterion(mask*warped, mask*target)# / torch.sqrt(mask[0,:,:,:].sum())
            return self.loss

def blind_denoising(**args):
    """
    Main function
    args: Parameters
    """

    ##########
    # LOAD THE DATA
    ##########

    np.random.seed(2019)
    if args['real']:
        print('Give sigma already normalized')
        sigma = args['sigma']
    else:
        sigma = args['sigma']/255

    sigma = min(max(sigma, 0.), 0.0784)


    model = load_net()
    model.cuda()

    dtype = torch.cuda.FloatTensor

    ########################
    # FINE-TUNING PARAMETERS
    ########################

    # Define loss
    lr = args['lr']
    weight_decay = 0.00001
    batch_size = 180

    criterion = WarpedLoss()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    #######################
    # MAIN LOOP FINE-TUNING
    #######################

    for idx1 in range(args['frames']-1, -1, -1):
        for idx2 in range(args['frames']):
            if idx1 != idx2:
                # read the two images
                im1 = piio.read(args['input'] % idx1).squeeze().astype(np.float)
                if len(im1.shape) < 4:
                    im1 = np.expand_dims(im1, 0)
                    im1 = np.expand_dims(im1, 0)
                    if args['real']:
                        im1 /= 65535.
                    else:
                        im1 /= 255.

                im1 = np.pad(im1, ((0,0),(0,0),(48,48),(48,48)), 'symmetric')

                im1 = torch.Tensor(im1).cuda().repeat(1, 3, 1, 1)
                curr_frame_var,_ = rgb2bayer(im1)

                im2 = piio.read(args['input'] % idx2).squeeze().astype(np.float)
                if len(im2.shape) < 4:
                    im2 = np.expand_dims(im2, 0)
                    im2 = np.expand_dims(im2, 0)
                    if args['real']:
                        im1 /= 65535.
                    else:
                        im1 /= 255.

                prev_frame_var = torch.Tensor(im2).cuda().repeat(1, 3, 1, 1)
                B,C,H,W = prev_frame_var.size()

                # read the transform
                p_file = open(args['input_p'] %(idx1,idx2), 'r')
                p = p_file.readline()
                p = p_file.readline()
                p = list(map(float, p.split(' ')[:-1]))

                sample = {"mosaic": curr_frame_var, "noise_level": sigma*torch.ones(1).cuda()}

                model.train()
                optimizer.zero_grad()

                # Do noise2noise1shot learning
                for it in range(args['iter']):
                    out_train = model(sample)
                    BO,CO,HO,WO = out_train.size()
                    DH = (HO-H) // 2 
                    DW = (WO-W) // 2 
                    loss = criterion(out_train[:,:,DH:-DH,DW:-DW], prev_frame_var, p)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    # Estimate the quality after overfitting
    noisy = piio.read(args['input'] % 0).squeeze().astype(np.float)
    if len(noisy.shape) < 4:
        noisy = np.expand_dims(noisy, 0)
        noisy = np.expand_dims(noisy, 0)
        if args['real']:
            im1 /= 65535.
        else:
            im1 /= 255.

    H = noisy.shape[2]
    W = noisy.shape[3]
    noisy = np.pad(noisy, ((0,0),(0,0),(48,48),(48,48)), 'symmetric')
    curr_frame_var,_ = rgb2bayer(Variable(torch.Tensor(noisy).cuda().repeat(1, 3, 1, 1)))
    sample = {"mosaic": curr_frame_var, "noise_level": sigma*torch.ones(1).cuda()}
    with torch.no_grad():  # PyTorch v0.4.0
        out = model(sample)
    BO,CO,HO,WO = out.size()
    DH = (HO-H) // 2 
    DW = (WO-W) // 2 
    out = out[:,:,DH:-DH,DW:-DW]
    out = out.cpu().numpy().transpose(2,3,1,0).squeeze().clip(0,1)

    if args['ref'] is not None:
        ref = piio.read(args['ref']).squeeze().astype(np.float) / 255.
        quant_psnr = psnr(ref, out)
        quant_ssim = compare_ssim(ref, out, data_range=1., multichannel=True)
        print(quant_psnr, quant_ssim)

    piio.write(args['output'], out)

    if args['output_network'] is not None:
        torch.save([model, optimizer], args['output_network'])


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Mosaic-to-mosaic fine-tuning")
    parser.add_argument("--input", type=str, default="",
                        help='path to input image')
    parser.add_argument("--input_p", type=str, default="",
                        help='path to input transform')
    parser.add_argument("--output_network", type=str,
                        help='path to output network')
    parser.add_argument("--iter", type=int, default=20,
                        help='number of time the learning is done on a given frame')
    parser.add_argument("--frames", type=int, default=10,
                        help='number of image in the burst')
    parser.add_argument("--net", type=str, default="/mnt/adisk/tehret/demosaicking/out/dmcnn_44_B2Bres.pth",
                        help='path to the network')
    parser.add_argument("--lr", type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument("--sigma", type=float, default=5,
                        help='sigma')
    parser.add_argument('--real', dest='real', action='store_true',
                        help='Used when processing real 16bits data')
    parser.add_argument("--output", type=str, default='out.tiff'
                        help='path to output image')
    parser.add_argument("--ref", type=str,
                        help='path to ref image')

    argspar = parser.parse_args()

    print("\n### Mosaic-to-mosaic fine-tuning ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    blind_denoising(**vars(argspar))
