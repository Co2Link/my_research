import numpy as np
import os
from keras.layers import Flatten, Conv2D, Input, Dense
from keras.models import Model
from atari_wrappers import wrap_deepmind, make_atari
import tensorflow as tf
from keras import backend as K
from collections import deque
from agents.base import Agent

MODEL_PATH = '../model/teacher'
GAME_Breakout = 'BreakoutNoFrameskip-v4'  # discrete(4)
GAME_Qbert = 'QbertNoFrameskip-v4'  # discrete(6)
GAME_Pong = 'PongNoFrameskip-v4'  # discrete(6)


class Teacher(Agent):
    """ teacher class which generate memory for student to learn """
    def __init__(self, path, env, epsilon=0.05, mem_size=50000):
        """ initiated by a trained model """
        self.env = env

        # load model
        model = self.build_CNN_model(input_shape=self.env.observation_space.shape, output_num=self.env.action_space.n)
        model.load_weights(path)
        self.model = model

        # Memory
        self.s_m = deque(maxlen=mem_size)  # State
        self.o_m = deque(maxlen=mem_size)  # output

        self.mem_gen = self._memory_generator()

        self.eps = epsilon

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
        output = self.model.predict_on_batch(np.expand_dims(state,axis=0))

        return np.argmax(output[0]), output[0]

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




def DEBUG():
    # keras setup
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
    sess = tf.Session(config=config)
    K.set_session(sess)

    env = make_atari(GAME_Breakout)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    # breakout_teacher = Teacher(os.path.join(MODEL_PATH, 'breakout.h5f'), GAME_Breakout)
    breakout_teacher = Teacher(os.path.join(MODEL_PATH, 'breakout.h5f'), env)

    breakout_teacher.play()

if __name__ == '__main__':
    DEBUG()
