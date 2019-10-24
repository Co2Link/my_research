import os
import time
import csv

from keras.callbacks import TensorBoard
import tensorflow as tf


class LogWriter():
    def __init__(self, root, batch_size, histogram_freq=0, write_graph=True, write_grads=False):

        self.root = root

        self.start_time = time.time()

        if not os.path.exists(self.root):
            os.mkdir(self.root)
            print('*** Create folder: {} ***'.format(self.root))

        now_time = time.strftime('%y%m%d_%H%M%S', time.localtime())
        self.save_path = os.path.join(self.root, now_time).replace('\\', '/')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            print('*** Create folder: {} ***'.format(self.save_path))

        os.mkdir(os.path.join(self.save_path, "logs").replace('\\', '/'))
        os.mkdir(os.path.join(self.save_path, "csv").replace('\\', '/'))
        os.mkdir(os.path.join(self.save_path, "models").replace('\\', '/'))
        os.mkdir(os.path.join(self.save_path, "movies").replace('\\', '/'))

        self.tb = TensorBoard(
            log_dir=os.path.join(self.save_path, "logs").replace('\\', '/'),
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=write_graph,
            write_grads=write_grads
        )

        # count batch
        self.batch_id = 0

        # minimum reward
        self.max_reward = -1e4

        # count iteration
        self.iteration = 1

    def get_movie_pass(self):
        return os.path.join(self.save_path, "movies").replace('\\', '/')

    def add_loss(self, losses):
        # log losses into tensorboard
        for loss, name in zip(losses, self.loss_names):
            summary = tf.Summary()
            summary.value.add(tag=name, simple_value=loss)
            self.tb.writer.add_summary(summary, self.batch_id)
            self.tb.writer.flush()
            self.tb.on_epoch_end(self.batch_id)

        # log losses into csv
        with open(os.path.join(self.save_path, 'csv', 'loss.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.batch_id, *losses])

        self.batch_id += 1

    def set_loss_name(self, names):
        """ set the first row for loss.csv """
        self.loss_names = names
        with open(os.path.join(self.save_path, 'csv', 'loss.csv').replace('\\', '/'), 'w', newline='') as f:
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

        with open(os.path.join(self.save_path, 'csv', 'max_reward.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            summary = tf.Summary()
            summary.value.add(tag="max_episode_reward",
                              simple_value=self.max_reward)
            for i in range(info['steps']):
                iteration = self.iteration - info['steps'] + i
                self.tb.writer.add_summary(summary, iteration)
                writer.writerow((iteration, self.max_reward))
            self.tb.writer.flush()

        # log episode_reward
        with open(os.path.join(self.save_path, 'csv', 'reward.csv').replace('\\', '/'), 'a', newline='') as f:

            # log episode_reward into tensorboard
            summary = tf.Summary()
            summary.value.add(tag="episode_reward",
                              simple_value=reward)  # change
            self.tb.writer.add_summary(summary, episode)
            self.tb.writer.flush()

            # log episode_reward into csv
            writer = csv.writer(f)
            writer.writerow((episode, reward))

    def count_iteration(self):
        """ count the iteration """
        self.iteration += 1

    def save_weights(self, agent, info):
        agent.save_weights(info, os.path.join(
            self.save_path, 'models').replace('\\', '/'))

    def save_model_arch(self, agent):
        agent.save_model_arch(os.path.join(
            self.save_path, 'models').replace('\\', '/'))

    def set_model(self, model):
        self.tb.set_model(model)

    def save_setting(self, args):
        with open(os.path.join(self.save_path, 'setting.csv').replace('\\', '/'), 'w', newline='') as f:
            writer = csv.writer(f)
            for k, v in vars(args).items():
                writer.writerow((k, v))
                print(k, v)

    def log_total_time_cost(self):
        """ Call it at the end of the code """
        with open(os.path.join(self.save_path, 'setting.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('total_time_cost', time.time() - self.start_time))
            print('*** total_time_cost:{} ***'.format(time.time() - self.start_time))
