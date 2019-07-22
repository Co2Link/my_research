import os
import numpy as np
import csv
from tqdm import tqdm
import glob

from agents.base import Agent


class Evaluator(Agent):
    """ used for evaluating student model """

    def __init__(self, env, log_path, epsilon=0.05, eval_iteration=10000):
        """
        Args:
            log_path: path to Log file . eg: "./result_DT/190721_195934"
        """
        super().__init__(env, None)
        self.env = env

        # path to model file
        model_path = glob.glob(os.path.join(log_path, 'models', '*'))[0].replace('\\', '/')
        # path to setting.csv file
        setting_path = glob.glob(os.path.join(log_path, 'setting.csv'))[0].replace('\\', '/')

        # load model
        model = self.build_CNN_model(input_shape=self.state_shape, output_num=self.action_num)
        model.load_weights(model_path)
        self.model = model

        # get the setting of epoch
        with open(setting_path, newline='') as f:
            reader = csv.reader(f)
            setting_dict = {row[0]: row[1] for row in reader}
        self.epoch = setting_dict['epoch']

        self.eps = epsilon

        self.iter = eval_iteration

        self.ep_rewards = []

        # get model_file_name from path
        self.model_file_name = model_path.split('/')[-1]

        # get log_file_name, remove the suffix of model_file_name
        model_name = ''.join(self.model_file_name.split('.')[0])
        self.log_file_name = 'reward_{}_{}.csv'.format(model_name, self.epoch)

        # log
        self.save_path = 'result_EVAL'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            print("*** make directory {} ***".format(self.save_path))

    def evaluate(self):

        print("*** Evaluating: {} ***".format(self.log_file_name))

        state = self.env.reset()

        ep_reward = 0

        for i in tqdm(range(self.iter),ascii=True):

            if self.eps <= np.random.uniform(0, 1):
                action = self.select_action(state)
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
        print("*** avg_ep_reward of {}: {} ***".format(self.log_file_name, avg_ep_reward))

        self.write_rewards()

        return avg_ep_reward

    def write_rewards(self):

        with open(os.path.join(self.save_path, self.log_file_name).replace('\\', '/'),
                  'w',
                  newline='') as f:
            writer = csv.writer(f)

            for ep_reward in self.ep_rewards:
                writer.writerow((ep_reward,))

    def select_action(self, state):
        state = self._LazyFrame2array(state)
        output = self.model.predict_on_batch(np.expand_dims(state, axis=0)).ravel()
        return np.argmax(output)

    def _LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)


if __name__ == '__main__':
    file_name = 'model_student_kld.h5f'

    print(file_name)
    a = file_name.split('.')[0]
    print(a)
    file_name = ''.join(a)
    print(file_name)
