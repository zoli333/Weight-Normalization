# Weight-Normalization-Complete

This is a complete Tensorflow implementation of Weight Normalization:

A Simple Reparameterization to Accelerate Training of Deep Neural Networks article
https://arxiv.org/pdf/1602.07868.pdf

Version: Tensorflow 1.5

The following methods are implemented and can be runned jointly or separately

  Markup : *initialization
  Markup : *Weight normalization
  


       * weight normalization with mean only batch normalization
         * batch normalization
         * standard parametrization (without initialization)

Batch normalization cannot be runned jointly with mean only batch normalization or weight normalization together. However weight normalization can be runned separately, or weight-norm + mean only batch norm can be runned jointly. The code also contains the standard parametrization without initialization.

The above methods can be defined in the run.sh bash script with the proper parameters set. One may also can add cuda visible devices to run on gpu.

The run.sh file contains an example run of weight-normalization with initialization and with mean-only-batch-normalization. This was reached the best accuracy in the article. 


Running:

   ./run.sh

