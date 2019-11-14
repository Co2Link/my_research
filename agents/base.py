import os
import time
import pickle

from abc import ABCMeta, abstractmethod
from collections import deque,namedtuple


Memory = namedtuple('Memory', ['state', 'action', 'reward', 'state_'])


class MemoryStorer:
    def __init__(self, size):
        self.memories = []
        self.size = size

    def add_memory_to_storation(self, s, a, r, ns):
        """ add memory for storation """
        self.memories.append(Memory(s, a, r, ns))

    def store_memories(self, path):
        """ save the memories as 'pkl' file """
        start_time = time.time()
        with open(os.path.join(path, 'memories.pkl'), 'wb') as f:
            pickle.dump(list(self.memories), f)
        print("*** time cost for storing memories(len: {}) {} ***".format(
            len(self.memories), time.time()-start_time))


class Agent(metaclass=ABCMeta):
    def __init__(self, state_shape, n_actions, logger):

        self.state_shape = state_shape

        self.n_actions = n_actions

        self.logger = logger

    @abstractmethod
    def select_action(self, state):
        pass
