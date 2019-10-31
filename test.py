
import time
import itertools
from collections import deque,namedtuple
import random
import numpy as np
from util.env_util import gather_numpy
from atari_wrappers import *
Memory = namedtuple('Memory',['state','action','reward','state_'])

env = make_atari("BreakoutNoFrameskip-v4")

print(type(env.action_space.sample()))
for i in range(100):
    print(random.randint(0,3))