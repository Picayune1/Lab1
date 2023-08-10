import torch
import numpy as np
import random
import matplotlib.pyplot as plt
# Device configuration
device = torch.device('cpu')
zs = torch.zeros((400,400))
#start at center
#use dimensions 800x and 800y with for loop of 6
#or use dimensions 400x and 400y with for loop of 5
zs[int(zs.size(dim=0)/2)][int(zs.size(dim=1)/2)] = 1
for i in range(5):
    up = torch.roll(zs,-1*(3**i),0)
    down = torch.roll(zs,1*(3**i),0)
    left = torch.roll(zs,-1*(3**i),1)
    right = torch.roll(zs,1*(3**i),1)
    zs = zs + up
    zs = zs + down
    zs = zs + left
    zs = zs + right
plt.imshow((zs).cpu().numpy())
plt.tight_layout()
plt.show()
