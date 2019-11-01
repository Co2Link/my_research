import numpy as np
import os
import csv
import time
import pickle
import random
from tqdm import tqdm
from atari_wrappers import wrap_deepmind, make_atari
import tensorflow as tf
from keras import backend as K
from collections import deque
from agents.base import Agent
from keras.models import model_from_json

MODEL_PATH = '../model/teacher'
GAME_Breakout = 'BreakoutNoFrameskip-v4'  # discrete(4)
GAME_Qbert = 'QbertNoFrameskip-v4'  # discrete(6)
GAME_Pong = 'PongNoFrameskip-v4'  # discrete(6)


class Teacher(Agent):
    """ teacher class which generate memory for student to learn """

    def __init__(self, model_path, env, epsilon=0.05, mem_size=50000, eval_iteration=10000):
        super().__init__(env, None)
        """ initiated by a trained model """
        self.env = env

        # load model
        with open(os.path.join(model_path,'models','model_arch.json'),'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(os.path.join(model_path,'models','model_weights_final.h5f'))

        # Memory
        self.s_m = deque(maxlen=mem_size)  # State
        self.o_m = deque(maxlen=mem_size)  # output

        self.mem_gen = self._memory_generator()

        self.eps = epsilon

        self.iter = eval_iteration

        self.ep_rewards = []

        # get model_file_name from model_path
        self.model_file_name = model_path.split('/')[-1]

        self.info = model_path

    def select_action(self, state):

        return self._select_action_output_logit(state)[0]

    def add_memories(self, size=5000):

        for i in range(size):
            state, output = next(self.mem_gen)
            self.s_m.append(state)
            self.o_m.append(output)
        print("*** add {} memories,memory size: {} ***".format(size, self.get_memory_size()))

    def get_memory_size(self):
        return len(self.s_m)

    def sample_memories(self, size=32):

        index = np.random.choice(len(self.s_m), size)
        index = list(index)

        s_batch = [self.s_m[ind] for ind in index]
        o_batch = [self.o_m[ind] for ind in index]

        return s_batch, o_batch

    def play(self):
        """ play the game to make sure model work fine """
        state = self.env.reset()

        while True:
            self.env.render()

            # select action
            if self.eps <= np.random.uniform(0, 1):
                action = self.select_action(state)
            else:
                action = self.env.action_space.sample()

            # step
            state_, _, done, _ = self.env.step(action)

            if done:
                state_ = self.env.reset()

            state = state_

    def _select_action_output_logit(self, state):
        """ return the selected action and the output logit """

        state = self._LazyFrame2array(state)
        output = self.model.predict_on_batch(np.expand_dims(state, axis=0)).ravel()

        return np.argmax(output), output

    def _memory_generator(self):

        state = self.env.reset()

        while True:

            if self.eps <= np.random.uniform(0, 1):
                action, output = self._select_action_output_logit(state)
                # generate memory
                yield state, output
            else:
                action = self.env.action_space.sample()

            # step
            state_, _, done, _ = self.env.step(action)

            # reset environment
            if done:
                state_ = self.env.reset()

            state = state_

    def _LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)

    def evaluate(self,save_path):
        print("*** Evaluating: {} ***".format(self.info))

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
        print("*** avg_ep_reward of {}: {} ***".format(self.info, avg_ep_reward))

        self.write_rewards(save_path)

        return avg_ep_reward

    def write_rewards(self,save_path):

        # get rid of the file suffix
        file_name = ''.join(self.model_file_name.split('.')[0])

        with open(os.path.join(save_path, 'reward_{}.csv'.format(file_name)).replace('\\', '/'),
                  'w',
                  newline='') as f:
            writer = csv.writer(f)

            for ep_reward in self.ep_rewards:
                writer.writerow((ep_reward,))

class Teacher_world_model(object):

    def __init__(self,agent_model_path,world_model_path,mem_size=100000):

        # load agent model
        with open(os.path.join(agent_model_path,'models','model_arch.json'),'r') as f:
            self.agent_model = model_from_json(f.read())
        self.agent_model.load_weights(os.path.join(agent_model_path,'models','model_weights_final.h5f'))
        
        # load world model
        with open(os.path.join(world_model_path,'models','model_arch.json'),'r') as f:
            self.world_model = model_from_json(f.read())
        self.world_model.load_weights(os.path.join(world_model_path,'models','model_weights_.h5f'))

        # load true memories
        with open(os.path.join(agent_model_path,'memories.pkl'),'rb') as f:
            self.t_m = pickle.load(f)

        # Memory
        self.i_m = [] # input  (n,84,84,4) 
        self.o_m = [] # output  (n,4)

        self.mem_gen = self._memory_generator()
        
        print('*** generating memories ***')
        for _ in tqdm(range(mem_size)):
            state,output = next(self.mem_gen)  # (84,84,4)  (4,)
            self.i_m.append(state)
            self.o_m.append(output)

    def _memory_generator(self):

        while True:
            state = random.sample(self.t_m,1)[0][0]
            state = self._LazyFrame2array(state)
            if np.array_equal(state,np.zeros(state.shape)):
                print('skip')
                continue
            for _ in range(100):

                if 0.05 <= np.random.uniform(0, 1):
                    
                    action, output = self._select_action_output_logit(state)
                    # generate memory
                    yield state, output
                else:
                    action = random.randint(0,3)

                # one-hot encoding
                one_hot_action = np.zeros((1,4))
                one_hot_action[0,action] = 1


                state_ = self.world_model.predict_on_batch(x = {'frames':np.expand_dims(self._LazyFrame2array(state),0),'actions':one_hot_action})

                state_ = np.concatenate((state[:,:,1:],state_.reshape((84,84,1))),axis=2)

                state = state_ 
    def sample_memories(self,size=32):
        index = np.random.choice(len(self.i_m), size)
        index = list(index)

        i_batch = [self.i_m[ind] for ind in index]
        o_batch = [self.o_m[ind] for ind in index]

        return i_batch, o_batch

    def _select_action_output_logit(self, state):
        """ return the selected action and the output logit """

        state = self._LazyFrame2array(state)
        output = self.agent_model.predict_on_batch(np.expand_dims(state, axis=0)).ravel()

        return np.argmax(output), output

    def _LazyFrame2array(self, lazyframe):
        return np.array(lazyframe).astype(np.float32) / 255.0

def DEBUG():
    # keras setup
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
    sess = tf.Session(config=config)
    K.set_session(sess)

    env = make_atari(GAME_Breakout)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    # breakout_teacher = Teacher(os.path.join(MODEL_PATH, 'breakout.h5f'), GAME_Breakout)
    breakout_teacher = Teacher(os.path.join(MODEL_PATH, 'breakout.h5f').replace('\\', '/'), env)

    breakout_teacher.play()


if __name__ == '__main__':
    DEBUG()
