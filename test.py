import os
import shutil
from collections import deque
import time
import tensorflow as tf
import numpy as np
import argparse
import csv
import gym

root='./root'

if __name__ == '__main__':
    env=gym.make('Breakout-v0')
    env.reset()

    action_space=env.action_space
    print(action_space)

    env=gym.make('QbertNoFrameskip-v4')
    env.reset()

    action_space=env.action_space
    print(action_space)

    env=gym.make('Breakout-v0')
    env.reset()

    action_space=env.action_space
    print(action_space)
