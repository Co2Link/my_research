from abc import ABCMeta, abstractmethod


class RL_runner(metaclass=ABCMeta):

    def __init__(self, logger=None):
        self.logger = logger

        self.episode = 0

    @abstractmethod
    def traj_generator(self, agent, env):
        pass

    @abstractmethod
    def train(self, agent, env, max_iter, batch_size, warmup=0, target_update_interval=0):
        pass

    @abstractmethod
    def test(self, agent, env, iter=1):
        pass
