import torch
import numpy as np
from torch.nn import Linear

inputs = torch.randn(1)

print(inputs)

linear1 = Linear(in_features=1, out_features=1)
linear1

print("W: ", linear1.weight)
print("B:", linear1.bias)

print("torch linear")
print(linear1(inputs))

print("python ile hesapladık")

print("x.w+b, m.inputs+b, w.girdi+b")
print(linear1.weight+inputs+linear1.bias)

print("------------------------------------------")

lin1 = Linear(in_features=1, out_features=5, bias=True)
print("Lin1 W:")
print(lin1.weight)
print("Li1 B:")
print(lin1.bias)

lin2 = Linear(in_features=5, out_features=8)
print("Lin2 W:")
print(lin2.weight)
print("Lin2 B:")
print(lin2.bias)
print("------------------------------------------")
print(lin2(lin1(inputs)))
