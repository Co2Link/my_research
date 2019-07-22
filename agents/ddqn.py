import random
import numpy as np

from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf

from util.ringbuf import RingBuf
from util.env_util import gather_numpy, scatter_numpy
from agents.base import Agent

import time


def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


class DDQN(Agent):
    """
    Double-Deep-Q-Learning
    almost the same implementation as the DQN-paper( Human-Level... )
    except this implementation has Double-Q-Learning and a different optimizer
    """

    def __init__(self, env, lr, gamma, logger, memory_size, batch_size, scale):
        super().__init__(env, logger)

        self.batch_size = batch_size

        # build network
        self.model = self.build_CNN_model(self.state_shape, self.action_num, "model")
        self.target = self.build_CNN_model(self.state_shape, self.action_num, "target")
        self.model.compile(optimizer=Adam(lr), loss=huberloss)
        self.target.compile(optimizer=Adam(lr), loss="mse")
        self.max_memory_size = int(memory_size)
        self.gamma = gamma
        self.scale = scale

        if logger is not None:
            logger.set_model(self.model)
            logger.set_loss_name([*self.model.metrics_names])

        # memory
        self.s_memory = RingBuf(size=self.max_memory_size)
        self.a_memory = RingBuf(size=self.max_memory_size)
        self.r_memory = RingBuf(size=self.max_memory_size)
        self.ns_memory = RingBuf(size=self.max_memory_size)

    def memorize(self, s, a, r, s_):
        """ Add memory """
        self.s_memory.append(s)
        self.a_memory.append(a)
        self.r_memory.append(r)
        self.ns_memory.append(s_)

    def select_action(self, state):
        state = self.LazyFrame2array(state)
        output = self.model.predict_on_batch(np.expand_dims(state, axis=0)).ravel()
        return np.argmax(output)

    def target_update(self):
        """ Update target network """
        self.target.set_weights(self.model.get_weights())

    def learn(self):
        """ Update model network """
        # sample from memory
        index = list(np.random.choice(len(self.s_memory), self.batch_size))

        s_batch = [self.s_memory[i] for i in index]
        a_batch = [self.a_memory[i] for i in index]
        r_batch = [self.r_memory[i] for i in index]
        ns_batch = [self.ns_memory[i] for i in index]

        s_batch = self.LazyFrame2array(s_batch)
        ns_batch = self.LazyFrame2array(ns_batch)

        # Compute
        q_model = self.model.predict_on_batch(s_batch)
        # choose action on next_state by model-network
        action_index = self.model.predict_on_batch(ns_batch).argmax(axis=1).reshape(-1, 1).astype(int)
        # compute corresponding action-value by target-network,
        q_target = self.target.predict_on_batch(ns_batch)
        q_target_sub = gather_numpy(q_target, 1, action_index) * self.gamma + np.array(r_batch).reshape(-1, 1)
        # scatter action-value on q_model
        q_model_ = scatter_numpy(np.copy(q_model), 1, np.array(a_batch).reshape(-1, 1).astype(int), q_target_sub)

        # # sanity check
        # print("q_model: \n", q_model[:3])
        # print("action_index: \n", action_index[:3])
        # print("q_target: \n", q_target[:3])
        # print("q_target_sub: \n", q_target_sub[:3])
        # print("q_model_: \n", q_model_[:3])
        # print("real_action_index: \n",np.array(a_batch).reshape(-1,1).astype(int)[:3])

        loss = self.model.train_on_batch(s_batch, q_model_)

        if self.logger is not None:
            self.logger.add_loss([loss])

        return loss

    def LazyFrame2array(self, LazyFrame):
        if self.scale:
            return np.array(LazyFrame)
        else:
            return np.array(LazyFrame).astype(np.float32) / 255.0

    def memory_size(self):
        return len(self.s_memory)
