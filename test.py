import torch
import numpy as np

a = torch.rand(4,4).to(device = 'cuda:0')

print(np.zeros(a[2,:].shape))