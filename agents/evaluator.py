import os
import numpy as np
import csv
from tqdm import tqdm
import glob
import time

from agents.base import Agent
from agents.ddqn import DDQN
from atari_wrappers import *


class Evaluator:

    def __init__(self, agent, env):

        self.env = env

        agent.model.eval()

        self.agent = agent

        self.ep_rewards = []

    def evaluate(self, epsilon=0.05, eval_iteration=500000, info=''):

        print("*** Evaluating{} ***".format(info))

        state = self.env.reset()

        ep_reward = 0

        for _ in tqdm(range(eval_iteration), ascii=True):

            if epsilon <= np.random.uniform(0, 1):
                action = self.agent.select_action(state)
            else:
                action = self.env.action_space.sample()

            state_, reward, done, _ = self.env.step(action)

            ep_reward += reward

            if done:
                state_ = self.env.reset()

                self.ep_rewards.append(ep_reward)
                ep_reward = 0

            state = state_

        avg_ep_reward = sum(self.ep_rewards) / len(self.ep_rewards)
        print("*** avg_ep_reward : {} ***".format(avg_ep_reward))

        return avg_ep_reward, self.ep_rewards

    def play(self, epsilon=0.05):
        state = self.env.reset()

        while True:
            time.sleep(0.01)
            self.env.render()

            if epsilon <= np.random.uniform(0, 1):
                action = self.agent.select_action(state)
            else:
                action = self.env.action_space.sample()

            state_, _, done, _ = self.env.step(action)

            if done:
                state_ = self.env.reset()

            state = state_


if __name__ == '__main__':
    pass