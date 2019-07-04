import numpy as np

from runners.rl_runner import RL_runner

class Normal_runner(RL_runner):
    def __init__(self,memory_size,eps_start=1.0,eps_end=0.01,eps_stop_rate=0.1,logger=None):
        super().__init__(logger)

