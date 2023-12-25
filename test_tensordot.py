import torch
import torch.nn as nn
import numpy as np

# https://stackoverflow.com/questions/41870228/understanding-tensordot

input_channels = 2
num_units = 1

input = torch.arange(90.).reshape(5, 2, 3, 3)
W = torch.arange(20.).reshape(10, 2)
c = torch.tensordot(W, input, dims=([1], [1]))
print(c.shape)
w = np.array(W)
x = np.array(input)
out_dim = (10,5,3,3)
el = np.zeros((out_dim))
for i in range(10):
    for j in range(5):
        for k in range(3):
            for q in range(3):
                for l in range(2):
                    el[i,j,k,q] += w[i, l] * x[j,l,k,q]
print(c.shape)
print(el.shape)

# out_r = c
# print(input.ndim)
# remaining_dims = range(2, input.ndim)
# print(remaining_dims)
# # bf01...
# print(out_r.shape)
# out = out_r.permute(1, 0, *remaining_dims)
# print(out.shape)
