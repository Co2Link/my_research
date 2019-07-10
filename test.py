import os
import shutil
from collections import deque
import time
import tensorflow as tf
import numpy as np
import argparse
import csv
import gym
from atari_wrappers import wrap_deepmind,make_atari

root='./root'

if __name__ == '__main__':
    env=make_atari('BreakoutNoFrameskip-v4')
    env=wrap_deepmind(env,frame_stack=True,scale=True)
    env.reset()

    action_space=env.action_space
    observation_space=env.observation_space
    print(action_space)
    print(observation_space.shape)
    print(type(observation_space.shape))

