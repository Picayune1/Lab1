import torch 
import numpy as np
import math
import matplotlib.pyplot as plt
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
# transfer to the GPU device
x = x.to(device)
y = y.to(device)
# Compute Gaussian
a = torch.exp(-(x**2+y**2)/2.0)
# Compute sine function
b = torch.sin(x-y)
z = a * b
plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()
