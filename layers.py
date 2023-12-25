import torch
import torch.nn as nn
from torch import linalg as LA


class WeightNormLayer(nn.Module):
    def __init__(self, module):
        super(WeightNormLayer, self).__init__()

        weight = getattr(module, 'weight')
        delattr(module, 'weight')

        module.register_parameter('weight_g', nn.Parameter(self.norm_except_dim_0(weight)))
        module.register_parameter('weight_v', nn.Parameter(weight))
        self.module = module

        self.compute_weight()

    def compute_weight(self):
        g = getattr(self.module, 'weight_g')
        v = getattr(self.module, 'weight_v')
        w = g * v / self.norm_except_dim_0(v)
        setattr(self.module, 'weight', w)

    @staticmethod
    def norm_except_dim_0(weight):
        output_size = (weight.size(0),) + (1,) * (weight.dim() - 1)
        out = LA.norm(weight.view(weight.size(0), -1), ord=2, dim=1).view(*output_size)
        return out

    def forward(self, x):
        self.compute_weight()
        return self.module.forward(x)


class MeanOnlyBatchNormLayer(nn.Module):
    def __init__(self, module):
        super(MeanOnlyBatchNormLayer, self).__init__()
        self.module = module

        if isinstance(self.module, WeightNormLayer):
            rootModule = self.module.module
        elif isinstance(self.module, (nn.Conv2d, nn.Linear, NINLayer)):
            rootModule = self.module
        else:
            self.module = None
            raise ValueError('Unsupported module:', module)

        weight = getattr(rootModule, 'weight')
        if getattr(rootModule, 'bias', None) is not None:
            delattr(rootModule, 'bias')
            rootModule.bias = None

        self.register_parameter('bias', nn.Parameter(torch.zeros((weight.size(0),), device=weight.device)))
        self.register_buffer('avg_batch_mean', torch.zeros(size=(weight.size(0),)))

    def forward(self, x):
        activation_prev = self.module.forward(x)
        output_size = (1,) + (activation_prev.size(1),) + (1,) * (activation_prev.dim() - 2)
        if not self.training:
            activation = activation_prev - self.avg_batch_mean.view(*output_size)
        else:
            num_outputs = activation_prev.size(1)
            mu = torch.mean(activation_prev.swapaxes(1, 0).contiguous().view(num_outputs, -1), dim=-1)
            activation = activation_prev - mu.view(*output_size)
            self.avg_batch_mean = 0.9 * self.avg_batch_mean + 0.1 * mu
        if hasattr(self, 'bias'):
            activation += self.bias.view(*output_size)
        return activation


class GaussianNoiseLayer(nn.Module):
    def __init__(self, device, sigma=0.15, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float32).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class NINLayer(nn.Module):
    def __init__(self, input_features, output_features,  activation=None, bias=True):
        super().__init__()
        self.num_units = output_features

        self.register_parameter('weight', nn.Parameter(torch.randn(output_features, input_features)))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(output_features,)))
        else:
            self.register_parameter('bias', None)

        self.apply(self._init_weights)

        self.activation = activation

    def _init_weights(self, module):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, x):
        out_r = torch.tensordot(self.weight, x, dims=([1], [1]))
        remaining_dims = range(2, x.ndim)
        out = out_r.permute(1, 0, *remaining_dims)

        if self.bias is not None:
            remaining_dims_biases = (1,) * (x.ndim - 2)  # broadcast
            b_shuffled = self.bias.view(1, -1, *remaining_dims_biases)
            out = out + b_shuffled
        if self.activation is not None:
            out = self.activation(out)
        return out


if __name__ == '__main__':
    m = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    w = MeanOnlyBatchNormLayer(WeightNormLayer(m))

