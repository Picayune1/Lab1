#Creating the Viscek cross
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
# Device configuration
device = torch.device('cpu')
cross = torch.zeros((2500,2500))
#start at center
#use dimensions 800x and 800y with for loop of 6
#or use dimensions 400x and 400y with for loop of 5
#or use dimensions 2500x and 2500y with for loop of 7
cross[int(cross.size(dim=0)/2)][int(cross.size(dim=1)/2)] = 1
for i in range(7):
    up = torch.roll(cross,-1*(3**i),0)
    down = torch.roll(cross,1*(3**i),0)
    left = torch.roll(cross,-1*(3**i),1)
    right = torch.roll(cross,1*(3**i),1)
    #utilising parrelism with pytorch
    cross = cross + up
    cross = cross + down
    cross = cross + left
    cross = cross + right
plt.imshow((cross).cpu().numpy())
plt.tight_layout()
plt.show()
