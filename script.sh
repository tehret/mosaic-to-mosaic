#!/bin/bash
MINV_PATH=../mInverse_bayer/inverse_compositional_algorithm

# Image 0 is always considered as the reference image
D=${1:-"burst_%02d.png"}
L=${2:-15}
N=${3:-0.02}


# Estimate the affinity between each pair of images in the burst
for i in `seq 0 $((L-1))`;
do
    for j in `seq 0 $((L-1))`;
    do
        $MINV_PATH `printf $D $j` `printf $D $i` -t 6 -f params_${i}_${j}.txt
    done
done

# Process the burst
python ../demosaicnet_torch/finetuning.py --input $D --input_p params_%d_%d.txt --frames $L --lr 1e-4 --iter 20 --sigma $N
