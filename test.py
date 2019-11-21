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


for i in range(100):
    print(random.randint(0,10))