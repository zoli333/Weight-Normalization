# Import necessary libraries
import torch
import torch.nn as nn
from layers import WeightNormLayer, MeanOnlyBatchNormLayer, GaussianNoiseLayer, NINLayer
import copy


#https://programming-review.com/pytorch/hooks
#https://arxiv.org/pdf/1906.02341.pdf
#https://github.com/victorcampos7/weightnorm-init/tree/master

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

layers = [
    GaussianNoiseLayer(device),
    nn.Conv2d(3, 96, kernel_size=3, padding=1),
    nn.Conv2d(96, 96, kernel_size=3, padding=1),
    nn.Conv2d(96, 96, kernel_size=3, padding=1),

    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=0.5),

    nn.Conv2d(96, 192, kernel_size=3, padding=1),
    nn.Conv2d(192, 192, kernel_size=3, padding=1),
    nn.Conv2d(192, 192, kernel_size=3, padding=1),

    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=0.5),

    nn.Conv2d(192, 192, kernel_size=3, padding=0),
    NINLayer(192, 192),
    NINLayer(192, 192),

    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(192, 10)
]


def data_dependent_weight_norm_g_init(wn_layer, sample_batch):
    """
    Data-dependent init of WN's g, as in Salimans & Kingma 2016.
        1) Set g=1, so that w = v / ||v||
        2) Set b = 0
        3) Obtain pre-activations:  y = w * x + b = v * x / ||v||
        4) Compute mu, sigma = mean(y), std(y)
        5) Initialize g = 1 / sigma, b = -mu / sigma
        6) Re-compute pre-activations with properly normalized layer
        7) Return the layer and the pre-activations, so that we can propagate the batch to the next layer
    """
    with torch.no_grad():
        wn_layer_module = wn_layer.module

        # Set g=1
        wn_layer_module.weight_g.uniform_(1, 1)

        # Set b=0
        wn_layer_module.bias.zero_()

        # Forward pass
        y = wn_layer(sample_batch)

        # Compute mean and std of pre-activations
        out_size = y.size(1)
        mu = torch.mean(y.swapaxes(1, 0).contiguous().view(out_size, -1), dim=-1)
        sigma = torch.sqrt(torch.mean(torch.square(y.swapaxes(1, 0).contiguous().view(out_size, -1) - mu.unsqueeze(1)), dim=-1))

        # Initialize parameters as in [Salimans & Kingma 2016] Eq (6)
        wn_layer_module.weight_g = nn.Parameter((1. / sigma).view(wn_layer_module.weight_g.size()))
        wn_layer_module.bias = nn.Parameter(-mu / sigma)

        # Re-compute pre-activations with properly normalized layer
        output_norm = wn_layer(sample_batch)
    return wn_layer, output_norm


def data_dependent_mean_only_batch_norm_g_init(mn_bn_layer, sample_batch):
    """
    Data-dependent init of WN's g, as in Salimans & Kingma 2016.
        1) Set g=1, so that w = v / ||v||
        2) Set b = 0
        3) Obtain pre-activations:  y = w * x + b = v * x / ||v||
        4) Compute mu, sigma = mean(y), std(y)
        5) Initialize g = 1 / sigma, b = -mu / sigma
        6) Re-compute pre-activations with properly normalized layer
        7) Return the layer and the pre-activations, so that we can propagate the batch to the next layer
    """
    # mn_bn_layer -> MeanOnlyBatchNormLayer(WeightNormLayer((..nn.Conv2d..))
    if isinstance(mn_bn_layer.module, (nn.Conv2d, nn.Linear, NINLayer)):
        raise ValueError('Unsupported module:', mn_bn_layer.module)

    with torch.no_grad():
        mn_bn_layer_module = mn_bn_layer.module.module

        # Set b=0
        mn_bn_layer.bias.zero_()

        # Set g=1
        mn_bn_layer_module.weight_g.uniform_(1, 1)

        # Forward pass
        # y = v*x/||v|| - mu (+ b)
        y = mn_bn_layer(sample_batch)

        # Compute std of pre-activations
        out_size = y.size(1)
        sigma = torch.sqrt(torch.mean(torch.square(y.swapaxes(1, 0).contiguous().view(out_size, -1)), dim=-1))

        # Initialize parameters as in [Salimans & Kingma 2016] Eq (6)
        mn_bn_layer_module.weight_g = nn.Parameter((1. / sigma).view(mn_bn_layer_module.weight_g.size()))

        # Re-compute pre-activations with properly normalized layer
        output_norm = mn_bn_layer(sample_batch)
    return mn_bn_layer, output_norm


def init_layer_maybe_normalize(layer, normalizer, init, sample_batch=None):
    if init in ['gaussian', 'gaussian_datadep']:
        torch.nn.init.normal_(layer.weight, 0.0, 0.05)
        torch.nn.init.zeros_(layer.bias)
    else:
        raise ValueError('Unsupported init:', init)

    if sample_batch is not None:
        sample_batch_ = sample_batch.detach().clone()

    if 'weight' in normalizer:
        layer = WeightNormLayer(layer).to(device)

        if 'datadep' in init:
            assert sample_batch_ is not None
            layer, sample_batch = data_dependent_weight_norm_g_init(layer, sample_batch_)

    if 'mean_only_batch_norm' in normalizer:
        layer = MeanOnlyBatchNormLayer(layer).to(device)

        if 'datadep' in init:
            assert sample_batch_ is not None
            layer, sample_batch = data_dependent_mean_only_batch_norm_g_init(layer, sample_batch_)
    elif 'batch_norm' in normalizer:
        num_outputs = layer.weight.size(0)
        if isinstance(layer, nn.Linear):
            layer = (layer, nn.BatchNorm1d(num_outputs))
        else:
            layer = (layer, nn.BatchNorm2d(num_outputs))

    return layer, sample_batch


class Model(nn.Module):
    def __init__(self, normalizer='no_norm', init='gaussian', sample_batch=None):
        super().__init__()
        self.layers = []
        layers_copy = copy.deepcopy(layers)
        last_layer_index = len(layers_copy) - 1

        for layer_idx, layer in enumerate(layers_copy):
            if isinstance(layer, (nn.Conv2d, nn.Linear, NINLayer)):
                layer, sample_batch = init_layer_maybe_normalize(layer, normalizer, init, sample_batch)

                if type(layer) == tuple:
                    [self.layers.append(l) for l in layer]
                else:
                    self.layers.append(layer)
                if layer_idx < last_layer_index:
                    leaky_relu = nn.LeakyReLU(negative_slope=0.1)
                    self.layers.append(leaky_relu)
                    if sample_batch is not None:
                        sample_batch = leaky_relu(sample_batch)
            else:
                self.layers.append(layer)
                if sample_batch is not None:
                    sample_batch = layer(sample_batch)

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    x = torch.randint(0, 255, size=(1,3,32,32)).float()
    y = (-127.5 + x) / 128.
    print(torch.min(y), torch.max(y))







