
import time
import itertools
from collections import deque,namedtuple
import random
import csv
import numpy as np
from util.env_util import gather_numpy
from atari_wrappers import *
from util.ringbuf import RingBuf
from tqdm import tqdm
Memory = namedtuple('Memory',['state','action','reward','state_'])

env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env,frame_stack=True,scale=True)


env.reset()

eps_rewards = []

eps_reward = 0

for i in tqdm(range(100000)):

    _,reward,done,_ = env.step(env.action_space.sample())

    eps_reward += reward

    if done:
        env.reset()
        eps_rewards.append(eps_reward)
        eps_reward = 0

print(eps_rewards)
print(sum(eps_rewards)/len(eps_rewards))
    