import torch
import numpy as np
from collections import namedtuple
from agents.base import MemoryStorer

Memory = namedtuple('Memory', ['state', 'action', 'reward', 'state_'])

a = MemoryStorer(123)

if a:
    print('adfas')