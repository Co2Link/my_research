import os

from keras import Sequential
from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model

from abc import ABCMeta, abstractmethod


class ModelBuilder:

    def build_FC_model(self, input_shape, output_num, name=""):
        """ build fully connected network """
        model = Sequential()
        model.add(Dense(10, activation="relu", name=(name + "_dense1"), input_shape=(None, input_shape)))
        model.add(Dense(10, activation="relu", name=(name + "_dense2")))
        model.add(Dense(10, activation="relu", name=(name + "_dense3")))
        model.add(Dense(output_num, activation="linear", name=(name + "q")))

        return model

    # expect input shape of (N,H,W,C)
    def build_CNN_model(self, input_shape, output_num, name="default"):
        """ build CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(512, activation="relu", name=(name + "_dense"))(x)

        q = Dense(output_num, activation='linear', name="q")(x)

        return Model(inputs, q)

    def save_model(self, info, file_name='model'):
        if os.path.exists(file_name + "_" + str(info) + ".h5f"):
            os.remove(file_name + "_" + str(info) + ".h5f")
        self.model.save_weights(file_name + "_" + str(info) + ".h5f")


class Agent(ModelBuilder, metaclass=ABCMeta):
    def __init__(self, env, logger):
        self.state_shape = env.observation_space.shape

        self.action_num = env.action_space.n

        print("state_shape: {}, action_num: {}".format(self.state_shape, self.action_num))

        self.logger = logger

        self.env = env

    @abstractmethod
    def select_action(self, state):
        pass
