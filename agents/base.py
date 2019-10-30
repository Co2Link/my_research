import os

from keras import Sequential
from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model

from abc import ABCMeta, abstractmethod


class ModelBuilder:

    def __init__(self):
        self.model_file_name = None
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

    def save_weights(self, info, path, file_name = 'model_weights'):
        """
        save model weights in .h5f file
        """
        path_with_file_name = path + '/' + file_name+ "_" + str(info)
        if os.path.exists(path_with_file_name + ".h5f"):
            os.remove(path_with_file_name + ".h5f")

        self.model.save_weights(path_with_file_name + ".h5f")

        self.model_file_name = (path_with_file_name + '.h5f').split('/')[-1]
    
    def save_model_arch(self,path,file_name = 'model_arch'):
        """
        save model architecture in json file
        """
        path_with_file_name = path + '/' + file_name
        if os.path.exists(path_with_file_name + '.json'):
            os.remove(path_with_file_name + '.json')
        with open(path_with_file_name + '.json','w') as json_file:
            json_file.write(self.model.to_json())
        


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
