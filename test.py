
import time
import itertools
from collections import deque

a=deque(maxlen=10)

for i in range(10):
    a.append(i)

print(a)
print(list(a))