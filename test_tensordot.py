import torch
import torch.nn as nn
import numpy as np

# https://stackoverflow.com/questions/41870228/understanding-tensordot


# Testing the first NINLayer
# -----------------------
# incoming activations from the conv layer before
# shape: (batch, out_features, width, height)
input = torch.randn((5, 2, 3, 3))
# NinLayer's weight shape (out_channels, in_features)
W = torch.arange(20.).reshape(10, 2)
# reduce dimensions along input's out_features axis and the weight matrix's in_features axis
c = torch.tensordot(W, input, dims=([1], [1]))
print(c.shape)
x = np.array(input)
w = np.array(W)
# out dim = (out_features, batch, width, height)
out_dim = (10, 5, 3, 3)
el = np.zeros((out_dim))
for i in range(10):
    for j in range(5):
        for k in range(3):
            for q in range(3):
                for l in range(2):
                    el[i,j,k,q] += w[i, l] * x[j,l,k,q]

print(c.shape)
print(el.shape)

# --- reshuffle the dimension of the output to match shape (batch, out_features, width, height)
out_r = c
#print(input.ndim)
remaining_dims = range(2, input.ndim)
#print(remaining_dims)
# bf01...
#print(out_r.shape)
out = out_r.permute(1, 0, *remaining_dims)
# (batch, out_features, width, height)
print(out.shape)


# Testing the second NINLayer
# -----------------------
# incoming activations from the NINLayer before
# shape: (batch, out_features, width, height)
input = out
# second NinLayer's weight shape (out_channels, in_features)
W = torch.randn((192, 10))
# reduce dimensions along input's out_features axis and the weight matrix's in_features axis
c = torch.tensordot(W, input, dims=([1], [1]))
print(c.shape)
w = np.array(W)
x = np.array(input)
out_dim = (192, 5, 3, 3)
el = np.zeros((out_dim))
for i in range(192):
    for j in range(5):
        for k in range(3):
            for q in range(3):
                for l in range(2):
                    el[i,j,k,q] += w[i, l] * x[j,l,k,q]

print(c.shape)
print(el.shape)

# --- reshuffle the dimension of the output to match shape (batch, out_features, width, height)
out_r = c
#print(input.ndim)
remaining_dims = range(2, input.ndim)
#print(remaining_dims)
# bf01...
#print(out_r.shape)
out = out_r.permute(1, 0, *remaining_dims)
# rearrange to have shape to have shape: (batch, out_features, width, height)
# (batch, out_features, width, height)
print(out.shape)
