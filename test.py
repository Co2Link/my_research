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

    s,r,d,_=env.step(env.action_space.sample())

    print(type(r))