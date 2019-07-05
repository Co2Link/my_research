import random

import numpy as np

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

from util.ringbuf import RingBuf

from agents.agent import Agent

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


class DDQN(Agent):
    def __init__(self,env,runner,lr,gamma,logger,memory_size,batch_size):
        super().__init__(env,logger)

        self.batch_size=batch_size

        # build network
        self.eval=self.build_CNN_model(self.state_shape,self.action_num,"eval")
        self.target=self.build_CNN_model(self.state_shape,self.action_num,"target")
        self.eval.compile(optimizer=Adam(lr),loss=huberloss)
        self.target.compile(optimizer=Adam(lr),loss="mse")
        self.memory_size=int(memory_size)

        if logger is not None:
            logger.set_model(self.eval)

            logger.set_loss_name = ([*self.eval.metrics_names])

        # memory

        self.s_memory=RingBuf(size=self.memory_size)
        self.a_memory=RingBuf(size=self.memory_size)
        self.r_memory=RingBuf(size=self.memory_size)
        self.ns_memory=RingBuf(size=self.memory_size)

    def memorize(self,s,a,r,s_):
        self.s_memory.append(s)
        self.a_memory.append(a)
        self.r_memory.append(r)
        self.ns_memory.append(s_)

    def select_action(self,state):
        state=self.LazyFrame2array(state)
        a=self.eval.predict_on_batch(np.array([state]))
        return np.argmax(a[0])

    def update_target_network(self):
        self.target.set_weights(self.eval.get_weights())

    def learn(self):
        index=list(np.random.choice(len(self.s_memory),self.batch_size))

        s_batch=[self.s_memory[i] for i in index]
        a_batch=[self.a_memory[i] for i in index]
        r_batch=[self.r_memory[i] for i in index]
        ns_batch=[self.ns_memory[i] for i in index]

        s_batch=self.LazyFrame2array(s_batch)
        ns_batch=self.LazyFrame2array(ns_batch)

        q_eval=self.eval.predict_on_batch(s_batch)

        action=self.eval.predict_on_batch(ns_batch).argmax(axis=1)
        q_target=self.target.predict_on_batch(ns_batch)






    def LazyFrame2array(self,LazyFrame):
        return np.array(LazyFrame)






