import os
import argparse
import math
import numpy as np
import torch
import iio as piio

# Generate noise
def sample_noise(size, sigma):
    return sigma*np.random.randn(size[0], size[1], size[2])

# Generate a transform
def sample_transform():
    return [1, (np.random.uniform(0, 1)-0.5)/20, 0, (np.random.uniform(0, 1)-0.5)/20, 1, 0, 0, 0, 1]

def rgb2bayer(im):
    H, W, C = im.shape
    bayer = np.zeros((H,W))

    # Red
    bayer[0::2, 1::2] = im[0::2, 1::2, 0]

    # green
    bayer[0::2, 0::2] = im[0::2, 0::2, 1]
    bayer[1::2, 1::2] = im[1::2, 1::2, 1]

    # blue
    bayer[1::2, 0::2] = im[1::2, 0::2, 2]

    return bayer

def cubic_interpolation(A,B,C,D,x):
    a,b,c,d = A.size()
    x = x.view(a,1,c,d).repeat(1,3,1,1)
    return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

# Return the img interpolated on the given grid, return also a mask
def bicubic_interpolation(im, grid):
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

    A = cubic_interpolation(im[:,:,y0,x0],im[:,:,y1,x0],im[:,:,y2,x0],im[:,:,y3,x0], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
    B = cubic_interpolation(im[:,:,y0,x1],im[:,:,y1,x1],im[:,:,y2,x1],im[:,:,y3,x1], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
    C = cubic_interpolation(im[:,:,y0,x2],im[:,:,y1,x2],im[:,:,y2,x2],im[:,:,y3,x2], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
    D = cubic_interpolation(im[:,:,y0,x3],im[:,:,y1,x3],im[:,:,y2,x3],im[:,:,y3,x3], grid[:,1,:,:] - torch.floor(grid[:,1,:,:]))
    
    output = cubic_interpolation(A,B,C,D,grid[:,0,:,:] - torch.floor(grid[:,0,:,:]))
    return output, mask
# Warp the image according to the transform p (p can be translation up to homography).
# It also returns a mask of unavailable pixels
def warp(x, p, scale=1):
    # Do the warping on the GPU because it's faster (and because I already had the code...)
    x = torch.Tensor(np.expand_dims(x, 0)).permute(0,3,1,2).cuda()

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, scale*W).view(1,-1).repeat(np.round(scale*H).astype(int),1)
    yy = torch.arange(0, scale*H).view(-1,1).repeat(1,np.round(scale*W).astype(int))
    xx = xx.view(1,1,np.round(scale*H).astype(int),np.round(scale*W).astype(int)).repeat(B,1,1,1).float()
    yy = yy.view(1,1,np.round(scale*H).astype(int),np.round(scale*W).astype(int)).repeat(B,1,1,1).float()

    # Center the new grid
    xx -= (scale-1)/2
    yy -= (scale-1)/2

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

    grid = torch.cat((xx_,yy_),1)
    vgrid = grid.cuda()
    output, mask = bicubic_interpolation(x, vgrid)
    return np.asarray(output[0].permute(1,2,0).cpu()), np.asarray(mask[0].permute(1,2,0).cpu())

def main(args):
    im = piio.read(args.im)

    # The first is always the reference image
    curr = im
    if args.sigma > 0:
        curr = curr + sample_noise(im.shape, args.sigma)
    if args.clip:
        curr = np.clip(curr, 0., 255.)
    curr_bayer = rgb2bayer(curr)
    piio.write(args.out % 0, curr_bayer)

    p_file = open(args.p, 'w')
    # The rest of the images are generated using warping according to a transform
    for i in range(1,args.len):
        p = sample_transform()
        p_file.write(str(p)+'\n')
        curr, _ = warp(im, p)
        if args.sigma > 0:
            curr = curr + sample_noise(im.shape, args.sigma)
        if args.clip:
            curr = np.clip(curr, 0., 255.)

        curr_bayer = rgb2bayer(curr)
        piio.write(args.out % i, curr_bayer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate a (noisy) burst')
    parser.add_argument('--im', type=str, default='image.png', help='Image used to generate the burst')
    parser.add_argument('--len', type=int, default=1, help='Length of the burst generated')
    parser.add_argument('--sigma', type=int, default=0, help='Noise standard deviation added to the burst')
    parser.add_argument('--out', type=str, default='burst_%d.tiff', help='Output burst')
    parser.add_argument('--p', type=str, default='transf.txt', help='File containing the transform used to generate the burst')
    parser.add_argument('--clip', type=bool, default=False, help='Clip the values of the generated burst')
    args = parser.parse_args()

    print(args)
    main(args)
