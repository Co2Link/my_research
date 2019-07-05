import os
import pathlib

import argparse

import gym
from gym import wrappers

from keras import backend as K
from keras.callbacks import TensorBoard

import tensorflow as tf
from agents.ddqn import DDQN
from runners.normal_runner import Normal_runner
#from runners.ale_atari_greedy_runner import ALE_Atari_greedy_runner

from atari_wrappers import *

# 学習用定数
MAX_ITERATION = 1000000
LEARNUNG_RATE = 0.001
BATCH_SIZE = 128
EPISODE = None
GAMMA = 0.99

EPS_START = 1.0
EPS_END = 0.001
EPS_STEP = 0.001

MAX_MEM_LEN = 10000
WARMUP = 1000
TARGET_UPDATE = 10000

ENV_NAME = 'CartPole-v0'

SAVE_MODEL_INTERVAL = 100

ROOT_PATH = "./root"

def ddqn_main():
    env=make_atari("BreakoutNoFrameskip-v4")

    env=wrap_deepmind(env,frame_stack=True,scale=True)

    runner=Normal_runner(100)

    ddqn_agent=DDQN(env,runner,LEARNUNG_RATE,GAMMA,None,1e4,32)

    tran_gen=runner.traj_generator(ddqn_agent,env)

    for i in tran_gen:
        print(i)

if __name__ == '__main__':
    ddqn_main()