"""Unit tests.   pytest -q test_layers.py     (or:  python test_layers.py)"""
import copy

import torch
import torch.nn as nn

from layers import WeightNormLayer, MeanOnlyBatchNormLayer, GaussianNoiseLayer, GlobalAvgPool2d, NINLayer
from model import (Model, device, init_layer_maybe_normalize, data_dependent_weight_norm_g_init,
                   data_dependent_mean_only_batch_norm_g_init)


def test_nin_layer_matches_einsum():
    torch.manual_seed(0)
    nin = NINLayer(10, 7)
    nn.init.normal_(nin.bias)  # non-zero bias, so the bias path is tested too
    x = torch.randn(5, 10, 3, 3)
    expected = torch.einsum('oi,bihw->bohw', nin.weight, x) + nin.bias.view(1, -1, 1, 1)
    out = nin(x)
    assert out.shape == (5, 7, 3, 3)
    assert torch.allclose(out, expected, atol=1e-5)


def test_global_avg_pool_matches_adaptive_pool():
    x = torch.randn(4, 6, 5, 5)
    ref = nn.Flatten()(nn.AdaptiveAvgPool2d((1, 1))(x))
    assert torch.allclose(GlobalAvgPool2d()(x), ref, atol=1e-6)


def test_weight_norm_reproduces_the_original_layer_at_init():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(4, 3, 8, 8)
    expected = conv(x)
    wn = WeightNormLayer(conv)  # g = ||v||  ->  w == v
    assert torch.allclose(wn(x), expected, atol=1e-5)
    # the effective weight has norm g for every output unit
    w = wn.module.weight.detach().reshape(8, -1)
    assert torch.allclose(w.norm(dim=1), wn.module.g.detach().reshape(8), atol=1e-5)


def test_weight_norm_gradients_reach_g_and_v():
    wn = WeightNormLayer(nn.Linear(5, 3))
    wn(torch.randn(4, 5)).sum().backward()
    assert wn.module.g.grad is not None and wn.module.g.grad is not None


def test_mean_only_bn_train_output_is_centered():
    torch.manual_seed(0)
    layer = MeanOnlyBatchNormLayer(WeightNormLayer(nn.Conv2d(3, 4, kernel_size=3, padding=1)))
    layer.train()
    out = layer(torch.randn(8, 3, 6, 6) + 2.0)
    assert torch.allclose(out.mean(dim=(0, 2, 3)), layer.bias.detach(), atol=1e-5)  # bias is 0 at init


def test_mean_only_bn_running_mean_has_no_autograd_graph():
    # regression test: the buffer used to keep the whole autograd graph alive (memory leak)
    layer = MeanOnlyBatchNormLayer(WeightNormLayer(nn.Conv2d(3, 4, kernel_size=3, padding=1)))
    layer.train()
    for _ in range(3):
        layer(torch.randn(8, 3, 6, 6))
    assert not layer.avg_batch_mean.requires_grad
    assert layer.avg_batch_mean.grad_fn is None
    assert layer.avg_batch_mean.abs().sum() > 0  # it was actually updated


def test_mean_only_bn_eval_uses_running_mean():
    torch.manual_seed(0)
    layer = MeanOnlyBatchNormLayer(WeightNormLayer(nn.Conv2d(3, 4, kernel_size=3, padding=1)))
    with torch.no_grad():
        layer.avg_batch_mean.copy_(torch.tensor([0.5, -1.0, 2.0, 0.0]))
        layer.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    layer.eval()
    x = torch.randn(2, 3, 6, 6)
    expected = layer.module(x) - layer.avg_batch_mean.view(1, -1, 1, 1) + layer.bias.view(1, -1, 1, 1)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_mean_only_bn_removes_the_wrapped_bias():
    conv = nn.Conv2d(3, 4, kernel_size=3)
    layer = MeanOnlyBatchNormLayer(WeightNormLayer(conv))
    assert layer.module.module.bias is None
    assert 'bias' in dict(layer.named_parameters())  # the layer's own bias


def test_weight_norm_layers_can_be_deep_copied():
    torch.manual_seed(0)
    layer = MeanOnlyBatchNormLayer(WeightNormLayer(nn.Conv2d(3, 4, kernel_size=3, padding=1)))
    layer.eval()
    x = torch.randn(2, 3, 6, 6)
    out = layer(x)
    clone = copy.deepcopy(layer)
    assert torch.allclose(clone(x), out, atol=1e-6)
    assert clone.module.module.g is not layer.module.module.g
    assert 'weight' in layer.module.module.__dict__  # the original still has its cached weight


def test_gaussian_noise_is_additive_and_train_only():
    torch.manual_seed(0)
    noise = GaussianNoiseLayer(sigma=0.15)
    x = torch.zeros(200000)
    noise.train()
    assert abs(noise(x).std().item() - 0.15) < 0.005  # a relative noise layer would give exactly 0 here
    noise.eval()
    assert torch.equal(noise(x), x)


def test_gaussian_noise_relative_mode_scales_with_the_input():
    torch.manual_seed(0)
    noise = GaussianNoiseLayer(sigma=0.15, relative=True)
    noise.train()
    assert torch.equal(noise(torch.zeros(100)), torch.zeros(100))


def _sample(n=128, c=3, size=16):
    torch.manual_seed(1)
    return torch.randn(n, c, size, size).to(device)


def test_weight_norm_data_dependent_init_gives_zero_mean_unit_std():
    torch.manual_seed(0)
    layer = WeightNormLayer(nn.Conv2d(3, 16, kernel_size=3, padding=1)).to(device)
    layer, out = data_dependent_weight_norm_g_init(layer, _sample())
    assert torch.allclose(out.mean(dim=(0, 2, 3)), torch.zeros(16, device=device), atol=1e-4)
    assert torch.allclose(out.std(dim=(0, 2, 3), unbiased=False), torch.ones(16, device=device), atol=1e-3)


def test_mean_only_bn_data_dependent_init_gives_unit_std():
    torch.manual_seed(0)
    layer = MeanOnlyBatchNormLayer(WeightNormLayer(nn.Conv2d(3, 16, kernel_size=3, padding=1))).to(device)
    layer, out = data_dependent_mean_only_batch_norm_g_init(layer, _sample())
    assert torch.allclose(out.mean(dim=(0, 2, 3)), torch.zeros(16, device=device), atol=1e-4)
    assert torch.allclose(out.std(dim=(0, 2, 3), unbiased=False), torch.ones(16, device=device), atol=1e-3)


def test_data_dependent_init_keeps_parameter_registration():
    layer = WeightNormLayer(nn.Conv2d(3, 8, kernel_size=3, padding=1)).to(device)
    g_before = layer.module.g
    layer, _ = data_dependent_weight_norm_g_init(layer, _sample())
    assert layer.module.g is g_before  # copied in place, not replaced
    assert sorted(n for n, _ in layer.named_parameters()) == ['module.bias', 'module.g', 'module.v']


def _all_normalizers():
    return [('no_norm', 'gaussian'), ('weight', 'gaussian'), ('weight', 'gaussian_datadep'),
            ('batch_norm', 'gaussian'), ('weight_mean_only_batch_norm', 'gaussian'),
            ('weight_mean_only_batch_norm', 'gaussian_datadep')]


def test_every_model_variant_builds_and_runs():
    torch.manual_seed(0)
    sample = torch.randn(16, 3, 32, 32).to(device)
    for normalizer, init in _all_normalizers():
        sb = sample if 'datadep' in init else None
        model = Model(normalizer=normalizer, init=init, sample_batch=sb).to(device)
        model.train()
        out = model(torch.randn(8, 3, 32, 32).to(device))
        assert out.shape == (8, 10), (normalizer, init)
        out.sum().backward()
        assert torch.isfinite(out).all(), (normalizer, init)


def test_init_forward_pass_does_not_pollute_the_running_means():
    sample = torch.randn(16, 3, 32, 32).to(device)
    model = Model('weight_mean_only_batch_norm', 'gaussian_datadep', sample_batch=sample)
    for m in model.modules():
        if isinstance(m, MeanOnlyBatchNormLayer):
            assert m.avg_batch_mean.abs().sum().item() == 0.0


def test_same_seed_gives_identical_initial_weights():
    sample = torch.randn(16, 3, 32, 32).to(device)

    def build():
        torch.manual_seed(7)
        return Model('weight', 'gaussian_datadep', sample_batch=sample.clone(), init_mode='eval').to(device)

    a, b = build(), build()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb), na


def test_whole_model_can_be_deep_copied():
    sample = torch.randn(16, 3, 32, 32).to(device)
    model = Model('weight_mean_only_batch_norm', 'gaussian_datadep', sample_batch=sample).to(device)
    model.eval()
    clone = copy.deepcopy(model)
    x = torch.randn(4, 3, 32, 32).to(device)
    assert torch.allclose(model(x), clone(x), atol=1e-5)


def test_plain_init_of_the_output_layer():
    sample = torch.randn(16, 3, 32, 32).to(device)
    model = Model('weight', 'gaussian_datadep', sample_batch=sample, datadep_last_layer=False)
    last = model.model[-1]  # the output layer: no activation after it
    g = last.module.g.detach().flatten()
    v_norm = last.module.v.detach().reshape(10, -1).norm(dim=1)
    assert torch.allclose(g, v_norm, atol=1e-5)  # plain WN init: g = ||v||
    assert last.module.bias.abs().sum().item() == 0.0


def test_datadep_init_without_weight_norm_is_rejected():
    try:
        init_layer_maybe_normalize(nn.Conv2d(3, 4, 3), 'batch_norm', 'gaussian_datadep', torch.randn(4, 3, 8, 8))
    except ValueError:
        return
    raise AssertionError('expected a ValueError')


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        fn()
        print('ok  ', name)
    print(f'{len(tests)} tests passed')
