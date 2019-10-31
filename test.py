
import time
import itertools
from collections import deque,namedtuple
import random
import csv
import numpy as np
from util.env_util import gather_numpy
from atari_wrappers import *
from util.ringbuf import RingBuf
Memory = namedtuple('Memory',['state','action','reward','state_'])


a = RingBuf(maxlen=10)

for i in range(10):
    a.append(i)

print(list(a))

