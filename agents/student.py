import numpy as np
from keras.layers import Flatten, Conv2D, Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from agents.agent import Agent, Student
import tensorflow as tf
from keras import backend as K
from keras.losses import kullback_leibler_divergence, mean_squared_error


def kullback_leibler_divergence(y_true, y_pred):
    tau = 0.01

    return K.sum(K.softmax(y_true / tau) * K.log(K.softmax(y_true / tau) / K.softmax(y_pred)), axis=-1)


class Student_():
    def __init__(self, teachers, logger):

        assert isinstance(teachers, list)

        self.teachers = teachers
        self.output_num = len(self.teachers)
        self.logger = logger

        self.model_dict = self.build_model()
        for teacher in self.teachers:
            self.model_dict[str(teacher)].compile(optimizer=Adam(lr=0.0001), loss='mse')

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
                s_batch = self.LazyFrame2array(s_batch)
                o_batch = np.array(o_batch)

                loss = self.model_dict[str(teacher)].train_on_batch(s_batch, o_batch)
                # self.logger.add_loss([loss])
            print(i)

        self.model_dict[str(teacher)].save_weights('student.h5f')

    def LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)


class SingleDtStudent(Student):
    """ distill from single teacher """

    def __init__(self, env, lr, logger, batch_size, epsilon, teacher, add_mem_num, update_num, epoch):
        super().__init__(env, logger)

        self.batch_size = batch_size

        self.eps = epsilon

        self.teacher = teacher

        # the amount of memory we add for each epoch
        self.add_mem_num = add_mem_num
        # the number of times of update for each epock
        self.update_num = update_num

        self.epoch = epoch

        # build model
        self.model = self.build_CNN_model(self.state_shape, self.action_num, "student")
        self.model.compile(optimizer=Adam(lr), loss=kullback_leibler_divergence)

        # set logger
        if logger is not None:
            logger.set_model(self.model)
            logger.set_loss_name([*self.model.metrics_names])

    def distill(self):

        for e in range(self.epoch):

            self.teacher.add_memories(size=self.add_mem_num)

            total_loss = 0
            for i in range(self.update_num):
                s_batch, o_batch = self.teacher.sample_memories()

                s_batch = self._LazyFrame2array(s_batch)
                o_batch = np.array(o_batch)

                loss = self.model.train_on_batch(s_batch, o_batch)

                total_loss += loss

                if self.logger is not None:
                    self.logger.add_loss([loss])

            print("*** Epoch:{} ,Memory-size: {},avg_loss: {} ***".format(e + 1, self.teacher.get_memory_size(),
                                                                          total_loss / self.update_num))

    def select_action(self, state):
        pass

    def _LazyFrame2array(self, LazyFrame):
        return np.array(LazyFrame)


if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0'))
    sess = tf.Session(config=config)
    K.set_session(sess)

    a=np.array([[1,2,3,4,5]]).astype(np.float32)
    b=K.softmax(a)
    c=sess.run(b)

    print(c)