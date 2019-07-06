import os
import shutil
from utils import RingBuf
from collections import deque
import time
import tensorflow as tf
import numpy as np

class myclass:
    def cout(self):
        if not hasattr(self,'my_list'):
            self.my_list=[]
        self.my_list.append(1)

    def printd(self):
        print(self.my_list)

a=myclass()

for i in range(10):
    a.cout()

a.printd()

