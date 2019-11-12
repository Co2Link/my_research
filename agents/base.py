import os
import time
import pickle

# from keras import Sequential
# from keras.layers import Dense, Input, Conv2D, Flatten
# from keras.models import Model
from tensorflow import keras
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense,Conv2D,Flatten

from abc import ABCMeta, abstractmethod
from collections import deque
from util.ringbuf import RingBuf


class MemoryStorer:
    def __init__(self, size):
        # RingBuf is much faster when size bigger than 100000
        self.memories_storation = RingBuf(maxlen=size)

    def add_memory_to_storation(self,memory):
        """ add memory for storation """
        self.memories_storation.append(memory)

    def store_memories(self, path):
        """ save the memories as 'pkl' file """
        start_time = time.time()
        with open(os.path.join(path, 'memories.pkl'), 'wb') as f:
            pickle.dump(list(self.memories_storation), f)
        print("*** time cost for storing memories(len: {}) {} ***".format(
            len(self.memories_storation), time.time()-start_time))

class ModelBuilder:

    def __init__(self):
        # self.model_file_name = None
        self.model = None

    def build_FC_model(self, input_shape, output_num, name=""):
        """ build fully connected network """
        model = Sequential()
        model.add(Dense(10, activation="relu", name=(
            name + "_dense1"), input_shape=(None, input_shape)))
        model.add(Dense(10, activation="relu", name=(name + "_dense2")))
        model.add(Dense(10, activation="relu", name=(name + "_dense3")))
        model.add(Dense(output_num, activation="linear", name=(name + "q")))

        return model

    # expect input shape of (N,H,W,C)
    def build_CNN_model(self, input_shape, output_num, name="default"):
        """ build CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4),
                   activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                   activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                   activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(512, activation="relu", name=(name + "_dense"))(x)

        q = Dense(output_num, activation='linear', name="q")(x)

        return Model(inputs, q)

    def build_big_CNN_model(self, input_shape, output_num, name="default"):
        """ build small CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=64, kernel_size=(8, 8), strides=(4, 4),
                   activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2),
                   activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                   activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(1024, activation="relu", name=(name + "_dense"))(x)

        q = Dense(output_num, activation='linear', name="q")(x)

        return Model(inputs, q)

        # expect input shape of (N,H,W,C)
    def build_small_CNN_model(self, input_shape, output_num, name="default"):
        """ build small CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4),
                   activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                   activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                   activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu", name=(name + "_dense"))(x)

        q = Dense(output_num, activation='linear', name="q")(x)

        return Model(inputs, q)
    def build_super_CNN_model(self, input_shape, output_num, name="default"):
        """ build small CNN network """
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=64, kernel_size=(8, 8), strides=(4, 4),
                   activation="relu", name=(name + "_conv2D_1"))(inputs)
        x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2),
                   activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                   activation="relu", name=(name + "_conv2D_3"))(x)

        x = Flatten()(x)
        x = Dense(2048, activation="relu", name=(name + "_dense"))(x)

        q = Dense(output_num, activation='linear', name="q")(x)

        return Model(inputs, q)
    def save_weights(self, path, info=''):
        """
        save model weights in .h5f file
        """
        assert isinstance(info,str),'info should be str'
        suffix = '_'+info if info else ''
        path_with_file_name = "{}/{}{}.h5f".format(path, 'model_weights', suffix)

        if os.path.exists(path_with_file_name):
            os.remove(path_with_file_name)

        self.model.save_weights(path_with_file_name)

    def save_model_arch(self, path, file_name='model_arch'):
        """
        save model architecture in json file
        """
        path_with_file_name = "{}/{}.json".format(path,file_name)

        if os.path.exists(path_with_file_name):
            os.remove(path_with_file_name)

        with open(path_with_file_name, 'w') as f:
            f.write(self.model.to_json())

class Agent(ModelBuilder, metaclass=ABCMeta):
    def __init__(self, env, logger):
        super().__init__()

        self.state_shape = env.observation_space.shape

        self.action_num = env.action_space.n

        self.logger = logger

        self.env = env

    @abstractmethod
    def select_action(self, state):
        pass
