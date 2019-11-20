import torch
import os
import numpy as np
import csv
from collections import namedtuple
from agents.base import MemoryStorer

from agents.evaluator import Evaluator
from agents.ddqn import DDQN
from atari_wrappers import *


a = np.random.rand(3,3)

print(a)
print(a.dtype)
print(type(np.float64))
print(isinstance(a,np.float64))