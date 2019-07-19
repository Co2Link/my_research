import numpy as np
from keras.layers import Flatten, Conv2D, Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from agents.agent import Agent
from atari_wrappers import wrap_deepmind, make_atari
import tensorflow as tf
from keras import backend as K


def kullback_leibler_divergence(y_true, y_pred):
    tau = 0.01
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum((y_true / tau) * K.log((y_true / tau) / y_pred), axis=-1)


class Student_():
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
            teacher.add_memories()
            print("memory: ", teacher.get_memory_size())

        for i in range(1000):
            teacher.add_memories()

            # 1000 time update every
            for m in range(1000):
                s_batch, o_batch, _ = teacher.sample_memories()
                s_batch=self.LazyFrame2array(s_batch)
                o_batch=np.array(o_batch)

                loss = self.model_dict[str(teacher)].train_on_batch(s_batch, o_batch)
                # self.logger.add_loss([loss])
            print(i)

        self.model_dict[str(teacher)].save_weights('student.h5f')

    def LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)


class Student(Agent):
    def __init__(self,env,lr,logger,batch_size,epsilon):
        super().__init__(env,logger)

        self.model=self.build_CNN_model(self.state_shape,self.action_num,"student")
        self.model.compile(optimizer=Adam(lr),loss=kullback_leibler_divergence)

        self.batch_size=batch_size

        self.eps=epsilon


    def learn(self,memories):
        s_batch,o_batch=memories

        s_batch=self._LazyFrame2array(s_batch)

        loss=self.model.train_on_batch(s_batch,o_batch)

        if self.logger is not None:
            self.logger.add_loss([loss])

    def select_action(self,state):
        pass

    def _LazyFrame2array(self, LazyFrame):
        return np.array(LazyFrame)


