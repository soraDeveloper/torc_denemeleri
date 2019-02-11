import numpy as np
import torch

"""""
#türev alma

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

print(x)
print(w)
print(b)

y = w * x + b

print(y)

y.backward()

print("dy/dw:", w.grad) #y=wx+b >  x = 3
print("dy/db:", b.grad) #y=wx+b > 1

"""""
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

print(inputs)
print(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

print(w)
print(b)

def model(x):
    return x @ w.t() +b

preds = model(inputs)

print(preds)

def mse(real,preds):
    diff = real-preds
    return torch.sum(diff*diff)/ diff.numel()

loss = mse(targets,preds)

print(loss)

loss.backward()
print("-----------------")
print(w)
print(w.grad)
print(b)
print(b.grad)

w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)

preds = model(inputs)
print(preds)
print("-------------")
loss = mse(targets,preds)
print(loss)

loss.backward()

with torch.no_grad():

    w -= w.grad*1e-5
    b -= b.grad*1e-5
    w.grad.zero_()
    b.grad.zero_()
print("-------")
print(w)

