# Import necessary libraries
import torch
import torch.nn as nn
from layers import WeightNormLayer, MeanOnlyBatchNormLayer, GaussianNoiseLayer, GlobalAvgPool2d, NINLayer

# https://programming-review.com/pytorch/hooks
# https://arxiv.org/pdf/1906.02341.pdf
# https://github.com/victorcampos7/weightnorm-init/tree/master

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPS = 1e-6  # guards the division by sigma in the data-dependent init (dead units)


def make_layers(noise_mode='additive'):
    """Fresh (uninitialized) layer list of the CIFAR-10 model of the paper (appendix)."""
    return [
        GaussianNoiseLayer(sigma=0.15, relative=(noise_mode == 'relative')),
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

        GlobalAvgPool2d(),
        nn.Linear(192, 10)
    ]


def _channel_stats(y):
    """Per-output-unit mean and (biased) std of pre-activations, over batch and spatial dims."""
    reduce_dims = [d for d in range(y.dim()) if d != 1]
    shape = (1, y.size(1)) + (1,) * (y.dim() - 2)
    mu = y.mean(dim=reduce_dims)
    sigma = torch.sqrt(torch.square(y - mu.view(shape)).mean(dim=reduce_dims))
    return mu, sigma.clamp_min(EPS)


def data_dependent_weight_norm_g_init(wn_layer, sample_batch):
    """
    Data-dependent init of WN's g and bias, as in Salimans & Kingma 2016.
        1) Set g=1, so that w = v / ||v||
        2) Set b = 0
        3) Obtain pre-activations:  y = w * x + b = v * x / ||v||
        4) Compute mu, sigma = mean(y), std(y)
        5) Initialize g = 1 / sigma, b = -mu / sigma
        6) Re-compute pre-activations with properly normalized layer
        7) Return the layer and the pre-activations, so that we can propagate the batch to the next layer
    """
    with torch.no_grad():
        root = wn_layer.module

        root.g.fill_(1.0)
        root.bias.zero_()

        y = wn_layer(sample_batch)
        mu, sigma = _channel_stats(y)

        # Eq (6) of the paper; in-place copies keep the Parameter objects (and their registration) intact
        root.g.copy_((1.0 / sigma).view_as(root.g))
        root.bias.copy_(-mu / sigma)

        output_norm = wn_layer(sample_batch)
    return wn_layer, output_norm


def data_dependent_mean_only_batch_norm_g_init(mn_bn_layer, sample_batch):
    """
    Data-dependent init of g for  MeanOnlyBatchNormLayer(WeightNormLayer(...)):
        1) g = 1, bias = 0
        2) y = v * x / ||v|| - mean(v * x / ||v||)        (the mean is removed by the layer itself)
        3) g = 1 / std(y)
        4) Re-compute the output and return it, so that the batch can be propagated to the next layer
    The bias stays 0: the layer's mean subtraction already centers the output.
    """
    if not isinstance(mn_bn_layer.module, WeightNormLayer):
        raise ValueError('Unsupported module:', mn_bn_layer.module)

    with torch.no_grad():
        root = mn_bn_layer.module.module

        mn_bn_layer.bias.zero_()
        root.g.fill_(1.0)

        y = mn_bn_layer(sample_batch)
        _, sigma = _channel_stats(y)

        root.g.copy_((1.0 / sigma).view_as(root.g))

        output_norm = mn_bn_layer(sample_batch)
    return mn_bn_layer, output_norm


def init_layer_maybe_normalize(layer, normalizer, init, sample_batch=None):
    if init not in ('gaussian', 'gaussian_datadep'):
        raise ValueError('Unsupported init:', init)

    nn.init.normal_(layer.weight, 0.0, 0.05)
    nn.init.zeros_(layer.bias)

    datadep = 'datadep' in init
    use_wn = 'weight' in normalizer
    use_mean_only_bn = 'mean_only_batch_norm' in normalizer
    use_bn = 'batch_norm' in normalizer and not use_mean_only_bn

    if datadep:
        if sample_batch is None:
            raise ValueError('Data-dependent init needs a sample batch.')
        if not use_wn:
            raise ValueError('Data-dependent init is only defined for weight normalized layers.')
        sample_batch = sample_batch.detach().clone()

    output = None  # pre-activations of the sample batch (only computed for data-dependent init)

    if use_wn:
        layer = WeightNormLayer(layer).to(device)
        # With mean-only BN the g-init is done below on the full (WN + mean-only BN) layer;
        # doing the plain WN init first would be redundant, since it is overwritten.
        if datadep and not use_mean_only_bn:
            layer, output = data_dependent_weight_norm_g_init(layer, sample_batch)

    if use_mean_only_bn:
        layer = MeanOnlyBatchNormLayer(layer).to(device)
        if datadep:
            layer, output = data_dependent_mean_only_batch_norm_g_init(layer, sample_batch)
    elif use_bn:
        num_outputs = layer.weight.size(0)
        if isinstance(layer, nn.Linear):
            layer = (layer, nn.BatchNorm1d(num_outputs))
        else:
            layer = (layer, nn.BatchNorm2d(num_outputs))

    return layer, output


class Model(nn.Module):
    """
    normalizer: 'no_norm' | 'weight' | 'batch_norm' | 'weight_mean_only_batch_norm'
    init:       'gaussian' | 'gaussian_datadep'  (data-dependent init needs sample_batch)
    noise_mode: 'additive' (as in the paper) | 'relative' (the old behaviour of this repo)
    init_mode:  'train' | 'eval' -- whether noise/dropout are active while the sample batch is propagated
                through the net during the data-dependent init.
    datadep_last_layer: False -> the output (logits) layer gets the plain weight-norm init (g = ||v||, b = 0)
                instead of the data-dependent one (which makes its g large, ~8, and the logits unit-variance).
    """

    def __init__(self, normalizer='no_norm', init='gaussian', sample_batch=None,
                 noise_mode='additive', init_mode='train', datadep_last_layer=True):
        super().__init__()
        if init_mode not in ('train', 'eval'):
            raise ValueError('Unsupported init_mode:', init_mode)

        layer_list = []
        raw_layers = make_layers(noise_mode)
        last_layer_index = len(raw_layers) - 1
        x = sample_batch  # propagated through the net layer by layer (data-dependent init only)

        for layer_idx, layer in enumerate(raw_layers):
            if isinstance(layer, (nn.Conv2d, nn.Linear, NINLayer)):
                layer_init = init
                if layer_idx == last_layer_index and not datadep_last_layer:
                    layer_init = 'gaussian'
                layer, x_out = init_layer_maybe_normalize(layer, normalizer, layer_init, x)

                if isinstance(layer, tuple):
                    layer_list.extend(layer)
                else:
                    layer_list.append(layer)
                if layer_idx < last_layer_index:
                    leaky_relu = nn.LeakyReLU(negative_slope=0.1)
                    layer_list.append(leaky_relu)
                    if x_out is not None:
                        x_out = leaky_relu(x_out)
                x = x_out
            else:
                layer_list.append(layer)
                if x is not None:
                    was_training = layer.training
                    if init_mode == 'eval':
                        layer.eval()
                    x = layer(x)
                    layer.train(was_training)

        self.model = nn.Sequential(*layer_list)

        # the init forward passes ran in training mode and updated the running means -> start from zero
        for m in self.modules():
            if isinstance(m, MeanOnlyBatchNormLayer):
                m.reset_running_stats()

    def forward(self, x):
        return self.model(x)
