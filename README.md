# Weight-Normalization-Complete

This is a complete Tensorflow implementation of Weight Normalization:

A Simple Reparameterization to Accelerate Training of Deep Neural Networks article
https://arxiv.org/pdf/1602.07868.pdf

Version: Tensorflow 1.5

The following methods are implemented and can be runned jointly or separately
  
       - Initialization with Weight normalization
       - Weight normalization without initalization
       - Initialization and Weight normalization with mean only batch normalization
       - Weight normalization with mean only batch normalization (without initialization)
       - Batch normalization (without initialization)
       - Standard parametrization (without initialization)

Batch normalization cannot be runned jointly with mean only batch normalization or weight normalization together. However weight normalization can be runned separately, or weight-norm + mean only batch norm can be runned jointly. The code also contains the standard parametrization without initialization.

The above methods can be defined in the run.sh bash script with the proper parameters set. One may also can add cuda visible devices to run on gpu.

The run.sh file contains an example run of weight-normalization with initialization and with mean-only-batch-normalization. This was reached the best accuracy in the article. 

To run the code:

    ./run.sh


Note: in the train.py code, after the initialization the samples are reshuffled. This was an intutitive thing. When initializing with an arbitrary number of images and we're setting up the parameters g, b, and with this the weights itself, we hope we catch the training set's most important features. Then we reshuffle the training examples and with this the already initialized parameters (g,b), and doing forward and backward steps on another set of examples (this is the reshuffle). 

This way I wanted to force the neural network to go with the initialized parameters at the first step, which I hoped catches the important features of the training set (by setting the parameters g,b...), and then see what it can do when other, now reshuffled batch of images are coming. Or in other words, can the initialization part catch the most important features from the training set by setting the parameters to these batches of images in the initialization pahse, and then training the parameters further with respect to the initialized parameters?
Mathematically speaking: could the parameters be depend on x, so g(x),b(x),V(x) could this be learned?

I haven't tested out this yet completely. But when I was running the code with and without initialization when not reshuffling the data after initialization, I got the same losses (because of the seed was set, I could make a comparison between these two cases). But when I reshuffled the data after initialization the loss was smaller than before, and it seemed to me it learned faster this way.



