# Weight-Normalization

This is a complete Pytorch implementation of Weight Normalization:

A Simple Reparameterization to Accelerate Training of Deep Neural Networks article
https://arxiv.org/pdf/1602.07868.pdf

Version: Pytorch 2.1.1+cu118

The following methods are implemented and can be runned jointly or separately

       - no normalization (default reference model)
       - Weight normalization with initialization
       - Weight normalization without initalization
       - Weight normalization with mean only batch normalization
       - Weight normalization with mean only batch normalization (with initialization)
       - Batch normalization (without initialization)

# Results

![train_plot.png](https://github.com/zoli333/Weight-Normalization/blob/master/train_plot.png)

![test_plot.png](https://github.com/zoli333/Weight-Normalization/blob/master/test_plot.png)

The "no normalization" model was trained with learning rate 0.003 got unstable behaviours during training.
Learning rate was reduced to 0.0003 to train "no normalization" model.

The weight normalization methods without initialization converged the fastest along with batch normalization.
However the best results achieved with weight normalization combined with mean only batch normalization and with initialization as per the article.
Overall, the models without initialization trained/optimized faster but the same models with initialization got the better accuracy.
- The training with weight normalization alone or mean only batchnorm combined with weight normalization were stable for different learning rates.
- Also less overfitting occured with weight normalization or weight normalization combined with mean only batch normalization during training.

Note: Mean only batch normalization without weight normalization omitted from the results since it became unstable at learning rate 0.003 similarly to the reference model (the reference model got unstable with learning rate 0.003 as well).
In this experiment mean only batch normalization without weight normalization applied before this layer like: MeanOnlyBatchNormLayer(...nn.Conv2d()...) for all layers that had trainable parameters: Linear, Conv2d, NINLayer.

# Model
The model was identical with the article's model, except ZCA whitening was not applied just a default transform at the beginning, moving the image pixel range to [-1, 1]

# Training
The deterministic behaviour was kept during training, generating the same examples in the same order during training and if initialization was present then the initial batch
of examples were deterministic during the initialization time.
The training was done with the command below in google colab for 100 epochs for each model:

!CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py

# References:

-  https://github.com/victorcampos7/weightnorm-init/tree/master
-  https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f
-  https://pytorch.org/docs/stable/_modules/torch/nn/utils/weight_norm.html#weight_norm
-  https://github.com/TimSalimans/weight_norm
-  https://github.com/openai/weightnorm
----------------------------------------
Gaussian noise layer taken from:
- https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/2

NINLayer rewritten from (lasagne):
- https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/dense.py#L127-L226
