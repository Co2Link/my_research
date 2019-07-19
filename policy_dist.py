import numpy as np
import os
from keras.layers import Flatten, Conv2D, Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from atari_wrappers import wrap_deepmind, make_atari
import tensorflow as tf
from keras import backend as K
from collections import deque
import time
from logWriter import LogWriter

MODEL_PATH = './teacher'
GAME_1 = 'BreakoutNoFrameskip-v4'  # discrete(4)
GAME_2 = 'QbertNoFrameskip-v4'  # discrete(6)
GAME_3 = 'PongNoFrameskip-v4'  # discrete(6)

MAX_MEM_SIZE = 50000
ADD_MEM_NUM = 5000
EPSILON = 0.05


class Teacher():
    def __init__(self, path, game_name, epsilon, max_mem_size, add_mem_num):
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

        # state
        self.s_m = deque(maxlen=max_mem_size)
        # output
        self.o_m = deque(maxlen=max_mem_size)
        # game_name
        self.g_m = deque(maxlen=max_mem_size)

        self.mem_gen = self.memory_generator()
        self.add_mem_num = add_mem_num

        self.epsilon = epsilon

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

    def select_action(self, state):
        state = self.LazyFrame2array(state)
        output = self.model.predict_on_batch(np.array([state]))

        # action and q value
        return np.argmax(output[0]), output[0]

    def memory_generator(self):

        state = self.env.reset()

        while True:

            if self.epsilon <= np.random.uniform(0, 1):
                action, output = self.select_action(state)
                # generate memory
                yield state, output, self.game_name
            else:
                action = self.env.action_space.sample()

            # step
            state_, _, done, _ = self.env.step(action)

            # reset environment
            if done:
                state_ = self.env.reset()

            state = state_

    def add_memory(self):
        # for i in range(self.add_mem_num):
        for i in range(500):
            state, output, game_name = next(self.mem_gen)
            self.s_m.append(state)
            self.o_m.append(output)
            self.g_m.append(game_name)

    def get_memory_size(self):
        return len(self.s_m)

    def sample_meory(self):
        index = np.random.choice(len(self.s_m), 32)
        index = list(index)

        s_m = [self.s_m[ind] for ind in index]
        o_m = [self.o_m[ind] for ind in index]
        g_m = [self.g_m[ind] for ind in index]

        return s_m, o_m, g_m

    def play(self):
        state = self.env.reset()

        while True:
            self.env.render()

            # select action
            if self.epsilon <= np.random.uniform(0, 1):
                action = self.select_action(state)
            else:
                action = self.env.action_space.sample()

            # step
            state_, reward, done, _ = self.env.step(action)

            if done:
                state_ = self.env.reset()

            state = state_

    def LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)

    def __str__(self):
        return self.game_name

def kullback_leibler_divergence(y_true, y_pred):
    tau = 0.01
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum((y_true / tau) * K.log((y_true / tau) / y_pred), axis=-1)


class Student():
    def __init__(self, teachers, logger):

        assert isinstance(teachers, list)

        self.teachers = teachers
        self.output_num = len(self.teachers)
        self.logger = logger

        self.model_dict = self.build_model()
        for teacher in self.teachers:
            self.model_dict[str(teacher)].compile(optimizer=Adam(lr=0.001), loss=kullback_leibler_divergence)

    def build_model(self, input_shape=(84, 84, 4), name="default"):
        """ build CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(512, activation="relu", name=(name + "_dense"))(x)

        # multiple output layer
        model_dict = {}
        for teacher in self.teachers:
            x_ = Dense(128, activation="relu", name=str(teacher) + "_dense_2")(x)
            q = Dense(teacher.output_num, activation='linear', name=str(teacher) + '_q')(x_)
            model_dict[str(teacher)] = Model(inputs, q)

        return model_dict

    def learn(self):

        self.logger.set_loss_name(['loss'])

        teacher = self.teachers[0]

        for i in range(0):
            teacher.add_memory()
            print("memory: ", teacher.get_memory_size())

        for i in range(1000):
            teacher.add_memory()

            # 1000 time update every
            for m in range(1000):
                s_batch, o_batch, _ = teacher.sample_meory()
                s_batch=self.LazyFrame2array(s_batch)
                o_batch=np.array(o_batch)

                loss = self.model_dict[str(teacher)].train_on_batch(s_batch, o_batch)
                # self.logger.add_loss([loss])
            print(i)

        self.model_dict[str(teacher)].save_weights('student.h5f')

    def LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)


def DEBUG():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
    sess = tf.Session(config=config)
    K.set_session(sess)

    breakout_teacher = Teacher(os.path.join(MODEL_PATH, 'breakout.h5f'), GAME_1, EPSILON, MAX_MEM_SIZE, ADD_MEM_NUM)
    # breakout_teacher.play()

    logger = LogWriter('./Distill', 32)

    student = Student([breakout_teacher], logger)

    student.learn()


def test():
    env = make_atari(GAME_1)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    print(env.action_space)


if __name__ == '__main__':
    DEBUG()
    # test()
