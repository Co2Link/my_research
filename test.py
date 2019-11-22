import torch
import os
import numpy as np
import csv
import random
from collections import namedtuple
from agents.base import MemoryStorer

from agents.evaluator import Evaluator
from agents.ddqn import DDQN
from atari_wrappers import *


a = torch.randn(2,3)

print(a)
a = torch.cat(a,dim=1)

print(a)