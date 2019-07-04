import os
import shutil
from utils import RingBuf
from collections import deque
import time

import numpy as np

my_deque=deque(maxlen=10000)

myRingBuf=RingBuf(size=10000)



for i in range(10000):
    a=np.random.rand(1)
    my_deque.append(a)
    myRingBuf.append(a)

mask = np.random.choice(len(myRingBuf), 32)
index = list(mask)

print(index)

time_1=time.time()

for i in range(1000000):
    a=[myRingBuf[i] for i in index]

time_2=time.time()
print("Ring_time: {}".format(time_2-time_1))

