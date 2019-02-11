import torch
import numpy as np
from sklearn.datasets import load_boston

x = torch.rand(1)
print(x)
print(x.size)
print(x.size())

temp = torch.FloatTensor([23, 24, 25, 23.4, 23, 546, 45])
print(temp)
print(temp.size())

bostonhouses = load_boston()
print(bostonhouses)

bostonhouses = torch.from_numpy(bostonhouses.data)
print(bostonhouses)

