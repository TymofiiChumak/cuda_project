# Hough transform visualization using NPP linrary

This program is designed to show straight line lookup on
images.

## Processing pipeline

### 0. Transform to grayscale

NPP implementation of Hough transform works on image with only
one channel. If color image was provided, it firstly should be
converted to grayscale.

### 1. Smoothing images by Gausian blur

In order to prepare image to applying edge detection algorithm
it needed to remove noise and flatten difference between neighbour
pixels, so there will be much less false-positive error in finding
edges at picture.

### 3. Find mean value of pixels

Thresholds for hysteresis used in Canny algorithm hloud be adjusted
to image for better accuracy. So make them depends on mean pixel value.

### 4. Canny edge detection

Hough transform requires image to be binarized with positive values in
regions of image that is age between arias with simillar color.

### 5. Hought transform

Apply final algorithm on prepared image. In output it produces coefs
for line equalities. Those line are drown on input image.

## Execution environment

Program requires presence of cuda libraries installled. Tested on host
with linux with kernel version 5.15.0-58, nvidia driver version 525.85.12
and CUDA version 12.0. For image processing it requires FreeImage present.

## Build and run

To recompile binaries and execute with test data run:

```bash
make clean build run
```

## Conclussions

Result of Hought transform work isn't acceptable in many cases. Test
images contains a lot of details that can interfere transformation
output. Also NPP implementation don't sort lines by their significance,
so many false positives are shown. Edge detection algorithm producess
mostly good results.
