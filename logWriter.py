import os
import time
import csv

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

        # count batch
        self.batch_id = 0

        # minimum reward
        self.max_reward = -1e4

        # count iteration
        self.iteration = 1

    def get_movie_pass(self):
        return os.path.join(self.root_dir_with_datetime, "movies").replace('\\', '/')

    def add_loss(self, losses):

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

        # log the max_episode_reward
        if self.max_reward < reward:
            self.max_reward = reward

        with open(os.path.join(self.root_dir_with_datetime, 'csv', 'max_reward.csv').replace('\\', '/'), 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(info['steps']):
                iteration = self.iteration - info['steps'] + i
                writer.writerow((iteration, self.max_reward))

        # log episode_reward
        with open(os.path.join(self.root_dir_with_datetime, 'csv', 'reward.csv').replace('\\', '/'), 'a', newline='') as f:

            # log episode_reward into csv
            writer = csv.writer(f)
            writer.writerow((episode, reward))

    def count_iteration(self):
        """ count the iteration """
        self.iteration += 1

    def save_model(self, agent, info = ''):
        agent.save_model(os.path.join(
            self.root_dir_with_datetime, 'models').replace('\\', '/'),info)

    def save_evaluate_rewards(self,evaluate_rewards):
        with open(os.path.join(self.root_dir_with_datetime,'evaluate_rewards.csv').replace('\\','/'),'w',newline='') as f:
            writer = csv.writer(f)

            # write the avg_eps_reward at the first line
            avg_eps_reward = sum(evaluate_rewards)/len(evaluate_rewards)

            writer.writerow((avg_eps_reward,))
            for reward in evaluate_rewards:
                writer.writerow((reward,))
            
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

    def store_memories(self,memory_storer):
        if memory_storer:
            memory_storer.store_memories(self.root_dir_with_datetime)

    def save_as_csv(self,data,filename):
        """
        Args:
            data: list of tuples(lists)
        """
        with open(os.path.join(self.root_dir_with_datetime,'csv',filename),'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)