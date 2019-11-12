import os
import time
import csv

from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf


class LogWriter():
    def __init__(self, root_dir, batch_size, histogram_freq=0, write_graph=True, write_grads=False):

        self.root_dir = root_dir

        self.start_time = time.time()

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
            print('*** Create folder: {} ***'.format(self.root_dir))

        now_time = time.strftime('%y%m%d_%H%M%S', time.localtime())
        self.root_dir_with_datetime = os.path.join(self.root_dir, now_time).replace('\\', '/')
        if not os.path.exists(self.root_dir_with_datetime):
            os.mkdir(self.root_dir_with_datetime)
            print('*** Create folder: {} ***'.format(self.root_dir_with_datetime))

        os.mkdir(os.path.join(self.root_dir_with_datetime, "logs").replace('\\', '/'))
        os.mkdir(os.path.join(self.root_dir_with_datetime, "csv").replace('\\', '/'))
        os.mkdir(os.path.join(self.root_dir_with_datetime, "models").replace('\\', '/'))
        os.mkdir(os.path.join(self.root_dir_with_datetime, "movies").replace('\\', '/'))

        self.tb = TensorBoard(
            log_dir=os.path.join(self.root_dir_with_datetime, "logs").replace('\\', '/'),
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=write_graph,
            write_grads=write_grads
        )

        self.tb_writer = tf.summary.create_file_writer(os.path.join(self.root_dir_with_datetime, "logs"))

        # count batch
        self.batch_id = 0

        # minimum reward
        self.max_reward = -1e4

        # count iteration
        self.iteration = 1

    def get_movie_pass(self):
        return os.path.join(self.root_dir_with_datetime, "movies").replace('\\', '/')

    def add_loss(self, losses):
        # log losses into tensorboard
        for loss, name in zip(losses, self.loss_names):
            with self.tb_writer.as_default():
                tf.summary.scalar(name,loss,step=self.batch_id)
            self.tb.on_epoch_end(self.batch_id)

        # log losses into csv
        with open(os.path.join(self.root_dir_with_datetime, 'csv', 'loss.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.batch_id, *losses])

        self.batch_id += 1

    def set_loss_name(self, names):
        """ set the first row for loss.csv """
        self.loss_names = names
        with open(os.path.join(self.root_dir_with_datetime, 'csv', 'loss.csv').replace('\\', '/'), 'w', newline='') as f:
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

        

        with open(os.path.join(self.root_dir_with_datetime, 'csv', 'max_reward.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            # summary = tf.Summary()
            # summary.value.add(tag="max_episode_reward",
            #                   simple_value=self.max_reward)
            for i in range(info['steps']):
                iteration = self.iteration - info['steps'] + i
                with self.tb_writer.as_default():
                    tf.summary.scalar('max_episode_reward',self.max_reward,step = iteration)
                writer.writerow((iteration, self.max_reward))

        # log episode_reward
        with open(os.path.join(self.root_dir_with_datetime, 'csv', 'reward.csv').replace('\\', '/'), 'a', newline='') as f:

            # log episode_reward into tensorboard

            with self.tb_writer.as_default():
                tf.summary.scalar('episode_reward',reward,step=episode)

            # log episode_reward into csv
            writer = csv.writer(f)
            writer.writerow((episode, reward))

    def count_iteration(self):
        """ count the iteration """
        self.iteration += 1

    def save_weights(self, agent, info = ''):
        agent.save_weights(os.path.join(
            self.root_dir_with_datetime, 'models').replace('\\', '/'),info)

    def save_model_arch(self, agent):
        agent.save_model_arch(os.path.join(
            self.root_dir_with_datetime, 'models').replace('\\', '/'))

    def save_evaluate_rewards(self,evaluate_rewards):
        with open(os.path.join(self.root_dir_with_datetime,'evaluate_rewards.csv').replace('\\','/'),'w',newline='') as f:
            writer = csv.writer(f)

            # write the avg_eps_reward at the first line
            avg_eps_reward = sum(evaluate_rewards)/len(evaluate_rewards)

            writer.writerow((avg_eps_reward,))
            for reward in evaluate_rewards:
                writer.writerow((reward,))
            
    def set_model(self, model):
        self.tb.set_model(model)

    def save_setting(self, args):
        with open(os.path.join(self.root_dir_with_datetime, 'setting.csv').replace('\\', '/'), 'w', newline='') as f:
            writer = csv.writer(f)
            for k, v in vars(args).items():
                writer.writerow((k, v))
                print(k, v)

    def log_total_time_cost(self):
        """ Call it at the end of the code """
        with open(os.path.join(self.root_dir_with_datetime, 'setting.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('total_time_cost', time.time() - self.start_time))
            print('*** total_time_cost:{} ***'.format(time.time() - self.start_time))

    def store_memories(self,agent):
        if agent.memory_storation_size:
            agent.store_memories(self.root_dir_with_datetime)
