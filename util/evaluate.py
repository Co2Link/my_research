import os
import numpy as np
import csv
from tqdm import tqdm


class Evaluator:
    """ used for evaluating agent """
    def __init__(self, env, agent, epsilon=0.05, iteration=10000):
        self.env = env

        self.agent = agent

        self.eps = epsilon

        self.iter = iteration

        self.ep_rewards = []

        # log
        self.save_path = 'result_EVAL'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            print("*** make directory {} ***".format(self.save_path))

    def evaluate(self):

        state = self.env.reset()

        ep_reward = 0

        for i in tqdm(range(self.iter)):

            if self.eps <= np.random.uniform(0, 1):
                action = self.agent.select_action(state)
            else:
                action = self.env.action_space.sample()

            state_, reward, done, _ = self.env.step(action)

            ep_reward += reward

            if done:
                state_ = self.env.reset()

                # log ep_reward
                self.ep_rewards.append(ep_reward)
                ep_reward = 0

            state = state_

        # return the average reward
        avg_ep_reward = sum(self.ep_rewards) / len(self.ep_rewards)
        print("*** avg_ep_reward of {}: {} ***".format(self.agent.model_file_name, avg_ep_reward))

        self.write_rewards()

        return avg_ep_reward

    def write_rewards(self):

        with open(os.path.join(self.save_path, 'reward_{}.csv'.format(self.agent.model_file_name)).replace('\\','/'), 'w', newline='') as f:
            writer = csv.writer(f)

            for ep_reward in self.ep_rewards:
                writer.writerow((ep_reward,))
