import random
import numpy as np
from collections import deque,namedtuple

from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf

from util.env_util import gather_numpy, scatter_numpy
from agents.base import Agent,MemoryStorer

import time
from tqdm import tqdm


def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)

Memory = namedtuple('Memory',['state','action','reward','state_'])

class DDQN(Agent,MemoryStorer):
    """
    Double-Deep-Q-Learning
    almost the same implementation as the DQN-paper( Human-Level... )
    except this implementation has Double-Q-Learning and a different optimizer
    """

    def __init__(self, env, lr, gamma, logger, memory_size, batch_size, scale, net_size, is_load_model,memory_size_storation):
        Agent.__init__(self, env, logger)
        MemoryStorer.__init__(self,memory_size_storation)

        self.batch_size = batch_size

        assert net_size in ['big','small','normal'],"net_size must be one of ['big','small','normal']"

        # build network
        if net_size == 'big':
            self.model = self.build_big_CNN_model(self.state_shape, self.action_num, "model")
            self.target = self.build_big_CNN_model(self.state_shape, self.action_num, "target")
        elif net_size == 'small':
            self.model = self.build_small_CNN_model(self.state_shape, self.action_num, "model")
            self.target = self.build_small_CNN_model(self.state_shape, self.action_num, "target")
        elif net_size == 'normal':
            self.model = self.build_CNN_model(self.state_shape, self.action_num, "model")
            self.target = self.build_CNN_model(self.state_shape, self.action_num, "target")

        if is_load_model:
            self.model.load_weights('./model/loaded_model.h5f')
        self.model.compile(optimizer=Adam(lr), loss=huberloss)
        self.target.compile(optimizer=Adam(lr), loss="mse")
        self.max_memory_size = int(memory_size)
        self.gamma = gamma
        self.scale = scale

        if logger is not None:
            logger.set_model(self.model)
            logger.set_loss_name([*self.model.metrics_names])

        # memory
        self.memories = deque(maxlen=self.max_memory_size)

    def memorize(self, s, a, r, s_):
        """ Add memory """
        self.memories.append(Memory(s,a,r,s_))
        self.memories_storation.append(Memory(s,a,r,s_))

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
        batch = random.sample(self.memories,self.batch_size)
        s_batch,a_batch,r_batch,ns_batch = zip(*batch)

        s_batch = self.LazyFrame2array(s_batch)
        ns_batch = self.LazyFrame2array(ns_batch)

        # Compute
        q_model = self.model.predict_on_batch(s_batch)
        # choose action on next_state by model-network
        action_index = self.model.predict_on_batch(ns_batch).argmax(axis=1).reshape(-1, 1).astype(int)
        # compute corresponding action-value by target-network,
        q_target = self.target.predict_on_batch(ns_batch)

        # q_targ = r when next_state == done
        for i in range(self.batch_size):
            if np.array_equal(ns_batch[i,:,:,:],np.zeros(ns_batch[i,:,:,:].shape)):
                q_target[i,:] = 0

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
        return len(self.memories)

    def evaluate(self,epsilon = 0.05,iter = 100000):

        print('*** Evaluating ***')

        state = self.env.reset()

        ep_reward = 0

        ep_rewards = []

        for _ in tqdm(range(iter), ascii=True):

            if epsilon <= np.random.uniform(0, 1):
                action = self.select_action(state)
            else:
                action = self.env.action_space.sample()

            state_, reward, done, _ = self.env.step(action)

            ep_reward += reward

            if done:
                state_ = self.env.reset()

                ep_rewards.append(ep_reward)
                ep_reward = 0
            
            state = state_

        avg_ep_reward = sum(ep_rewards) / len(ep_rewards)
        print("*** avg_ep_reward of {} ***".format(avg_ep_reward))

        return ep_rewards
