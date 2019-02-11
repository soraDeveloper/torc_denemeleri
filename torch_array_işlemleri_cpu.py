import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

ördek = np.array(Image.open("ördek.JPG").resize((224, 224)))

ördek_torch = torch.from_numpy(ördek)

print(np.shape(ördek))
print(ördek_torch.size())

plt.imsave("ördek2", ördek_torch[50:150, 125:225, :])
plt.imsave("ördek3", ördek_torch[50:260, 125:225, :])