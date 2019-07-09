import os
import shutil
import time

import csv

from keras.callbacks import TensorBoard

import tensorflow as tf


class LogWriter():
    def __init__(self, root, batch_size, histogram_freq=0, write_graph=True, write_grads=False):
        # root path for all log file
        self.root = root

        self.start_time = time.time()

        count=0
        while(os.path.exists(self.root)):
            self.root=root+'-'+str(count)
            count+=1

        os.mkdir(self.root)
        os.mkdir(os.path.join(self.root, "logs"))
        os.mkdir(os.path.join(self.root, "csv"))
        os.mkdir(os.path.join(self.root, "models"))
        os.mkdir(os.path.join(self.root, "movies"))

        self.tb = TensorBoard(
            log_dir=os.path.join(self.root, "logs"),
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=write_graph,
            write_grads=write_grads
        )

        self.batch_id = 0

        self.names = None

        self.max_reward = -1e4 # minimum reward
        self.iteration = 1

    def get_movie_pass(self):
        return os.path.join(self.root, "movies")

    def add_loss(self, losses):

        # log losses into tensorboard
        for loss, name in zip(losses, self.loss_names):
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=loss)
            self.tb.writer.add_summary(summary, self.batch_id)
            self.tb.writer.flush()
            self.tb.on_epoch_end(self.batch_id)

        # log losses into csv
        with open(os.path.join(self.root, 'csv', 'loss.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.batch_id, *losses])

        self.batch_id += 1

    def set_loss_name(self, names):
        """ set the first row for loss.csv """
        self.loss_names = names
        with open(os.path.join(self.root, 'csv', 'loss.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.loss_names)

    def add_reward(self, episode, reward, info={}):
        """ record episode_reward and max_episode_reward """

        # Standard output
        print(episode, ":", reward, end="")

        for key in info.keys():
            print(", ", key, ":", info[key], end="")

        print()

        # log the max_episode_reward
        if self.max_reward < reward:
            self.max_reward = reward

        with open(os.path.join(self.root,'csv','max_reward.csv'),'a',newline='') as f:
            writer=csv.writer(f)
            summary = tf.Summary()
            summary.value.add(tag="max_episode_reward", simple_value=self.max_reward)
            for i in range(info['steps']):
                iteration=self.iteration - info['steps'] + i + 1
                self.tb.writer.add_summary(summary, iteration)
                writer.writerow((iteration,self.max_reward))
            self.tb.writer.flush()

        # log episode_reward
        with open(os.path.join(self.root, 'csv', 'reward.csv'), 'a', newline='') as f:

            # log episode_reward into tensorboard
            summary = tf.Summary()
            summary.value.add(tag="episode_reward", simple_value=reward)  # change
            self.tb.writer.add_summary(summary, episode)
            self.tb.writer.flush()

            # log episode_reward into csv
            writer = csv.writer(f)
            writer.writerow((episode, reward))

    def count_iteration(self):
        """ count the iteration """
        self.iteration += 1

    def save_model(self, agent, episode):
        agent.save_model(episode, os.path.join(self.root, 'models', 'model'))

    def set_model(self, model):
        self.tb.set_model(model)

    def save_setting(self, args):
        with open(os.path.join(self.root, 'setting.csv'), 'w',newline='') as f:
            writer = csv.writer(f)
            for k, v in vars(args).items():
                writer.writerow((k, v))
                print(k,v)

    def log_total_time_cost(self):
        """ Call it at the end of the code """
        with open(os.path.join(self.root, 'setting.csv'), 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('total_time_cost', time.time() - self.start_time))
            print('total_time_cost: ',time.time()-self.start_time)

