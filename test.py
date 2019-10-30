
import time
import itertools
from collections import deque,namedtuple
import random
import numpy as np
from util.env_util import gather_numpy
Memory = namedtuple('Memory',['state','action','reward','state_'])

a = np.random.rand(5,4)

b = np.array([[0],[1],[2],[3],[0]])
print(a)
print(b)
print(b.shape)

print(gather_numpy(a,1,b))
