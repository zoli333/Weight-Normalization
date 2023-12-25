# Weight-Normalization

This is a complete Tensorflow implementation of Weight Normalization:

A Simple Reparameterization to Accelerate Training of Deep Neural Networks article
https://arxiv.org/pdf/1602.07868.pdf

Version: Pytorch 2.1.1+cu118

The following methods are implemented and can be runned jointly or separately

       - no normalization (default with learning rate 0.0003, other learning rate settings got unstable behaviours during training)
       - Weight normalization with initialization
       - Weight normalization without initalization
       - Weight normalization with mean only batch normalization
       - Weight normalization with mean only batch normalization (with initialization)
       - Batch normalization (without initialization)

References:

-  https://github.com/victorcampos7/weightnorm-init/tree/master
-  https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f
-  https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html#weight_norm

