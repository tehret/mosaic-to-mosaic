Original repository available [here](https://github.com/gfacciol/demosaicnet_torch)

Pytorch implementation (using pretrained weights) of:

     Deep Joint Demosaicking and Denoising, SiGGRAPH Asia 2016.
     Michaël Gharbi, Gaurav Chaurasia, Sylvain Paris, Frédo Durand

This code (based on <https://github.com/mgharbi/demosaicnet_torch> and <https://github.com/mgharbi/demosaicnet_caffe>) was created to [reproduce the results of demosaicnet for this publication](https://doi.org/10.5201/ipol.2019.274)

    T. Ehret, and G. Facciolo, "A Study of Two CNN Demosaicking Algorithms", 
    Image Processing On Line, 9 (2019), pp. 220–230. https://doi.org/10.5201/ipol.2019.274

Example call (result is saved as output.png):

     python3 ./demosaicnet.py --input data/36111627_7991189675.jpg --output output.png --noise 0.01 

The training script has been adapted to be run out of the box (visualization tools have been removed).
(Note that while the training script has been tested to see if it was running without crashing, it has
not been tested until convergence!)

Example call for the training (output and checkpoints are saved in the `output` folder):

    python3 ./train.py data/training/dataset.txt output

Note that the dataset should be given as a list of images from the same folder. A random 128x128 patch is extracted
from them if the images are larger than this size (it possible to have duplicates inside the list to force multiple
sampling per image per epoch).
