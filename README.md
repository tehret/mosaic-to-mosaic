IMPLEMENTATION OF JOINT DEMOSAICKING AND DENOISING BY FINE-TUNING OF BURSTS OF RAW IMAGES
=========================================================================================

* Author    : EHRET Thibaud <ehret.thibaud@gmail.com>
* Licence   : AGPL v3+

OVERVIEW
--------

This code is provided to reproduce results from
 "Joint Demosaicking and Denoising by Fine-Tuning of Bursts of Raw Images
, T. Ehret, A. Davy, P. Arias, G. Facciolo, ICCV 2019".
Plase cite it if you use this code as part of your research.

USAGE
-----

The registration algorithm has to be compiled first (see the part about the modified inverse
compositional algorithm).

We provide two scripts, one to reproduce the results on synthetic data (generated noisy bayer
burst from a single reference image) and one to reproduce the results on real data coming from the HDR+ dataset.

In order to reproduce the synthetic results of the paper, the `finetuning_toy.sh` script has to be used.

```finetuning_toy.sh image.png 10 5```

There are three input arguments:
* the path to the reference image (C format). It is assumed that it is an RGB image with a range between 0 and 255.
* the number of images to be generated for the burst.
* the noise level to be added (between 0 and 255).

An example of command on one of the image of the kodak dataset:

```finetuning_toy.sh data/kodak/kodim19.png 10 5```


In order to reproduce the results of the supplementary material, the `finetuning_real.sh` script has to be used.

```finetuning_real.sh burst_%d.tiff 7 0.03```

There are three input arguments:
* the path to the burst (C format), index 0 should always be the reference image. It is assumed that the images are 16bits (not normalized).
* the number of images in the burst.
* the noise level (already normalized between 0 and 1).

In our case, the noise levels were estimated using "Miguel Colom, and Antoni Buades, Analysis and Extension of the Ponomarenko et al.
Method, Estimating a Noise Curve from a Single Image, Image Processing On Line, 3 (2013), pp. 173–197"

The four commands to reproduce the supplementary material HDR+ results:
* ```finetuning_real.sh data/hdrplus/1/%d.tiff 9 0.015```
* ```finetuning_real.sh data/hdrplus/2/%d.tiff 9 0.015```
* ```finetuning_real.sh data/hdrplus/3/%d.tiff 7 0.03```
* ```finetuning_real.sh data/hdrplus/4/%d.tiff 5 0.076```



MODIFIED INVERSE COMPOSITIONAL ALGORITHM
----------------------------------------

The code used for the registration for the ICCV paper is provided in the `mInverse_bayer` folder. It's a modified version of 
"Thibaud Briand, Gabriele Facciolo, and Javier Sánchez, Improvements of the Inverse Compositional Algorithm for Parametric
 Motion Estimation, Image Processing On Line, 8 (2018), pp. 435–464"

The code is compilable on Unix/Linux and hopefully on Mac OS (not tested!).

**Compilation:** requires the cmake and make programs.

Compile the source code using make.

UNIX/LINUX/MAC:
```
$ make
```

Binaries will be created in the current repository.
