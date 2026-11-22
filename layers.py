import copy
import numpy as np
import torch
import torch.nn as nn
from torch import linalg as LA


class WeightNormLayer(nn.Module):
    """Weight normalization: w = g * v / ||v||  (the norm is taken per output unit).

    Wraps an nn.Conv2d / nn.Linear / NINLayer. Its 'weight' parameter is replaced by
    'weight_g' (magnitude) and 'weight_v' (direction); the effective weight is recomputed
    from them at every forward pass.
    """


    def __init__(self, module):
        super().__init__()

        weight = getattr(module, 'weight')
        delattr(module, 'weight')

        # .detach(): g must be a leaf parameter and must not carry autograd history
        module.register_parameter('g', nn.Parameter(self.norm_except_dim_0(weight).detach()))
        module.register_parameter('v', nn.Parameter(weight.detach()))
        self.module = module

        self.compute_weight()

    def compute_weight(self):
        g = getattr(self.module, 'g')
        v = getattr(self.module, 'v')
        w = g * v / self.norm_except_dim_0(v)
        # plain (non-leaf) tensor attribute, not a Parameter -> it is not part of state_dict()
        setattr(self.module, 'weight', w)

    @staticmethod
    def norm_except_dim_0(weight):
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        # reshape (not view): also works for non-contiguous weights
        out = LA.norm(weight.reshape(weight.size(0), -1), ord=2, dim=1).view(*output_size)
        return out

    def forward(self, x):
        self.compute_weight()
        return self.module(x)

    def __deepcopy__(self, memo):
        # The cached 'weight' is a non-leaf tensor, and PyTorch refuses to deepcopy those
        # ("Only Tensors created explicitly by the user support the deepcopy protocol").
        # So: take it out, copy everything else, put it back, and rebuild it on the copy.
        cached_weight = self.module.__dict__.pop('weight', None)
        try:
            cls = self.__class__
            new = cls.__new__(cls)
            memo[id(self)] = new
            for key, value in self.__dict__.items():
                new.__dict__[key] = copy.deepcopy(value, memo)
        finally:
            if cached_weight is not None:
                self.module.__dict__['weight'] = cached_weight
        new.compute_weight()
        return new


class MeanOnlyBatchNormLayer(nn.Module):
    """Mean-only batch normalization (Salimans & Kingma 2016, section 4):
       subtract the batch mean of the pre-activations, then add a learnable bias.
       The wrapped module's own bias is removed (this layer's bias replaces it).
       """

    def __init__(self, module, momentum=0.1):
        super().__init__()

        if isinstance(module, WeightNormLayer):
            root = module.module
            weight = root.g
        elif isinstance(module, (nn.Conv2d, nn.Linear, NINLayer)):
            root = module
            weight = root.weight
        else:
            raise ValueError('Unsupported module:', module)

        self.module = module
        self.momentum = momentum

        if getattr(root, 'bias', None) is not None:
            delattr(root, 'bias')
            root.bias = None

        num_outputs = weight.size(0)
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_outputs, device=weight.device)))
        self.register_buffer('avg_batch_mean', torch.zeros(num_outputs, device=weight.device))

    def reset_running_stats(self):
        self.avg_batch_mean.zero_()

    def forward(self, x):
        out = self.module(x)
        shape = (1, out.size(1)) + (1,) * (out.dim() - 2)
        if self.training:
            reduce_dims = [d for d in range(out.dim()) if d != 1]
            mu = out.mean(dim=reduce_dims)
            # The running mean must NOT keep an autograd graph alive -> update under no_grad / detached.
            with torch.no_grad():
                self.avg_batch_mean.mul_(1 - self.momentum).add_(mu.detach(), alpha=self.momentum)
            centered = out - mu.view(shape)
        else:
            centered = out - self.avg_batch_mean.view(shape)
        return centered + self.bias.view(shape)


class GaussianNoiseLayer(nn.Module):
    """Gaussian noise, active only in training mode.

    relative=False (default): additive noise  x + N(0, sigma^2)   (as in the paper / Lasagne's GaussianNoiseLayer)
    relative=True           : noise scaled by the value itself      x + N(0, 1) * sigma * x
    """

    def __init__(self, sigma=0.15, relative=False):
        super().__init__()
        self.sigma = sigma
        self.relative = relative

    def forward(self, x):
        if self.training and self.sigma != 0:
            noise = torch.randn_like(x) * self.sigma
            if self.relative:
                noise = noise * x.detach()
            x = x + noise
        return x


class GlobalAvgPool2d(nn.Module):
    """Global average pooling, (N, C, H, W) -> (N, C).

    Same result as nn.AdaptiveAvgPool2d((1, 1)) + nn.Flatten(), but its backward pass is deterministic on CUDA
    (AdaptiveAvgPool2d's backward is not), which makes torch.use_deterministic_algorithms(True) usable.
    """

    def forward(self, x):
        return x.mean(dim=(2, 3))


class NINLayer(nn.Module):
    """Network-in-network (1x1 convolution) layer, rewritten from Lasagne."""

    def __init__(self, input_features, output_features, activation=None, bias=True):
        super().__init__()
        self.num_units = output_features

        self.register_parameter('weight', nn.Parameter(torch.randn(output_features, input_features)))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(output_features, )))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

        self.activation = activation

    def forward(self, x):
        # weight: (out, in), x: (batch, in, H, W) -> (out, batch, H, W) -> (batch, out, H, W)
        out_r = torch.tensordot(self.weight, x, dims=([1], [1]))
        remaining_dims = range(2, x.ndim)
        out = out_r.permute(1, 0, *remaining_dims)

        if self.bias is not None:
            remaining_dims_biases = (1,) * (x.ndim - 2)  # broadcast
            out = out + self.bias.view(1, -1, *remaining_dims_biases)
        if self.activation is not None:
            out = self.activation(out)
        return out


if __name__ == '__main__':
    m = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    w = MeanOnlyBatchNormLayer(WeightNormLayer(m))

