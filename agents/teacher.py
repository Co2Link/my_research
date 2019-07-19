import numpy as np
import os
from keras.layers import Flatten, Conv2D, Input, Dense
from keras.models import Model
from atari_wrappers import wrap_deepmind, make_atari
import tensorflow as tf
from keras import backend as K
from collections import deque

MODEL_PATH = '../teacher'
GAME_Breakout = 'BreakoutNoFrameskip-v4'  # discrete(4)
GAME_Qbert = 'QbertNoFrameskip-v4'  # discrete(6)
GAME_Pong = 'PongNoFrameskip-v4'  # discrete(6)



class Teacher():
    def __init__(self, path, game_name, epsilon=0.05, max_mem_size=50000):

        # make environment
        env = make_atari(game_name)
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        self.env = env
        self.game_name = game_name
        self.output_num = self.env.action_space.n

        # load model
        model = self.build_model(input_shape=self.env.observation_space.shape, output_num=self.output_num)
        model.load_weights(path)
        self.model = model

        # Memory
        self.s_m = deque(maxlen=max_mem_size)   # State
        self.o_m = deque(maxlen=max_mem_size)   # output

        self.mem_gen = self._memory_generator()

        self.eps = epsilon

    def build_model(self, input_shape, output_num, name="defalut"):
        """ build CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(512, activation="relu", name=(name + "_dense"))(x)

        q = Dense(output_num, activation='linear', name="q")(x)

        return Model(inputs, q)

    def select_action(self,state):
        return self._select_action_output_logit(state)[0]

    def add_memories(self,size=5000):

        for i in range(size):
            state, output = next(self.mem_gen)
            self.s_m.append(state)
            self.o_m.append(output)
        print("* add {} memories,memory size: {} *".format(size,self.get_memory_size()))

    def get_memory_size(self):
        return len(self.s_m)

    def sample_memories(self,size=32):

        index = np.random.choice(len(self.s_m), size)
        index = list(index)

        s_batch = [self.s_m[ind] for ind in index]
        o_batch = [self.o_m[ind] for ind in index]

        return s_batch, o_batch

    def play(self):
        state = self.env.reset()

        while True:
            self.env.render()

            # select action
            if self.eps <= np.random.uniform(0, 1):
                action = self.select_action(state)
            else:
                action = self.env.action_space.sample()

            # step
            state_, reward, done, _ = self.env.step(action)

            if done:
                state_ = self.env.reset()

            state = state_

    def _select_action_output_logit(self, state):
        """ return the selected action and the output logit """

        state = self._LazyFrame2array(state)
        output = self.model.predict_on_batch(np.array([state]))

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

    def __str__(self):
        return self.game_name

def DEBUG():
    # keras setup
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
    sess = tf.Session(config=config)
    K.set_session(sess)

    breakout_teacher=Teacher(os.path.join(MODEL_PATH,'breakout.h5f'),GAME_Breakout)
    breakout_teacher.play()

if __name__ == '__main__':
    DEBUG()