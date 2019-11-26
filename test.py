import torch
import os
import numpy as np
import csv
import random
from collections import namedtuple
from agents.base import MemoryStorer
import matplotlib.pyplot as plt

from agents.evaluator import Evaluator
from agents.ddqn import DDQN
from atari_wrappers import *

from agents.teacher_ import Wrapper_sp

hparams = {'n_action':4,}

model_path = 'result_WORLD/191125_161456/models/model_5.pt'

memory_path = 'result/191119_214214/memories.pkl'

env =  Wrapper_sp(hparams=hparams,memory_path=memory_path,model_path=model_path)

state = env.reset()


fig,ax = plt.subplots(1,7)
ax[0].imshow(state[:,:,-1],interpolation='nearest')

next_state = env.step(2)

ax[1].imshow(next_state[:,:,-1],interpolation='nearest')
next_state = env.step(2)

ax[2].imshow(next_state[:,:,-1],interpolation='nearest')
next_state = env.step(2)

ax[3].imshow(next_state[:,:,-1],interpolation='nearest')
next_state = env.step(2)

ax[4].imshow(next_state[:,:,-1],interpolation='nearest')

next_state = env.reset()

ax[5].imshow(next_state[:,:,-1],interpolation='nearest')

next_state = env.step(2)

ax[6].imshow(next_state[:,:,-1],interpolation='nearest')




plt.show()

