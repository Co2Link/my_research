
import time
import itertools
from collections import deque,namedtuple
import random
import numpy as np


Memory = namedtuple('Memory',['state','action','reward','state_'])

a = np.random.rand(2,3)

print(a)

for i in a:
    print(i)