import torch

rand1 = torch.rand(1000,1000)
rand2 = torch.rand(1000,1000)

#cpu kısmı
""""
print("----------")
#print(rand1+rand2)
print("----------")
print(rand1.mul(rand2))
print(rand1,rand2)
print("----------")
print(rand1*rand2)
print("----------")
print(rand1+rand2)
print("----------")
print(rand1.mul(rand2))"""
# gpu kısmı için
rand1 = rand1.cuda()
rand2 = rand2.cuda()

