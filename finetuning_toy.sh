#!/bin/bash
MINV_PATH=mInverse_bayer/inverse_compositional_algorithm

# Image 0 is always considered as the reference image
D=${1:-"kodim19.png"}
L=${2:-10}
N=${3:-5}

# Generate a toy burst
python data/generate_burst.py --im $D --len $L --sigma $N

# Estimate the affinity between each pair of images in the burst
for i in `seq 0 $((L-1))`;
do
    for j in `seq 0 $((L-1))`;
    do
        $MINV_PATH `printf $D $j` `printf $D $i` -t 6 -f params_${i}_${j}.txt
    done
done

# Process the reference image without fine-tuning
python demosaicnet_torch/demosaicnet.py --input burst_0.tiff --output gharbi.tiff --noise $N

# Process the burst
python demosaicnet_torch/finetuning.py --input burst_%d.tiff --input_p params_%d_%d.txt --frames $L --lr 1e-4 --iter 20 --sigma $N

# Compute PSNR + SSIM
python assert_quality --input gharbi.tiff --ref $D
python assert_quality --input gharbi_finetuned.tiff --ref $D

# Clean things up
rm *.txt
