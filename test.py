import torch
import os
import numpy as np
import csv
from collections import namedtuple
from agents.base import MemoryStorer

from agents.evaluator import Evaluator
from agents.ddqn import DDQN
from atari_wrappers import *


a = np.append([1, 2, 3], [4, 5, 6, 7, 8, 9])

print(a)