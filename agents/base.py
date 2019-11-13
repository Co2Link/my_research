import os
import time
import pickle

from abc import ABCMeta, abstractmethod
from collections import deque
from util.ringbuf import RingBuf


class MemoryStorer:
    def __init__(self, size):
        # RingBuf is much faster when size bigger than 100000
        self.memories_storation = RingBuf(maxlen=size)

    def add_memory_to_storation(self, memory):
        """ add memory for storation """
        self.memories_storation.append(memory)

    def store_memories(self, path):
        """ save the memories as 'pkl' file """
        start_time = time.time()
        with open(os.path.join(path, 'memories.pkl'), 'wb') as f:
            pickle.dump(list(self.memories_storation), f)
        print("*** time cost for storing memories(len: {}) {} ***".format(
            len(self.memories_storation), time.time()-start_time))


class Agent(metaclass=ABCMeta):
    def __init__(self, state_shape, n_actions, logger):

        self.state_shape = state_shape

        self.n_actions = n_actions

        self.logger = logger

    @abstractmethod
    def select_action(self, state):
        pass
