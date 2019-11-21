import gym
import os
import time
import numpy as np
import random
import pickle
import csv
from collections import deque,namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl_networks import ACVP


# from agents.teacher import Teacher_world_model
# from agents.student import SingleDtStudent_world

import torch

from atari_wrappers import *
from logWriter import LogWriter

from util.decorators import timethis

Memory = namedtuple('Memory',['state','action','reward','state_'])

class Memory_generator:
    def __init__(self,root_path):

        self.root_path = root_path

        self.action_space_size = 4

        self.memories = []

        self._restore_memories()

    @timethis
    def _restore_memories(self):

        with open(os.path.join(self.root_path,'memories.pkl'),'rb') as f:
            memories = pickle.load(f)
        self.zero_array = np.zeros(np.shape(memories[0][0]))
        
        # 9:1 for training and testing
        self.train_memories = memories[int(len(memories)/10):]
        self.test_memories = memories[:int(len(memories)/10)]
        print('*** traning size: {},testing size: {}'.format(len(self.train_memories),len(self.test_memories)))

    def sample_batches(self,batch_size,is_test = False,prediction_steps=1):
        states,one_hot_actions,rewards,state_s = [],[],[],[]
        for _ in range(batch_size):
            state,one_hot_action,reward,state_ = self._sample_memories_MultiStep(prediction_steps,is_test)
            states.append(state)
            one_hot_actions.append(one_hot_action)
            rewards.append(reward)
            state_s.append(state_)

        return np.array(states),

    def sample_memories(self,batch_size,is_test = False):
        """ sample memories for trainning or testing

        Shape:
            states: (N,84,84,4)
            one_hot_actions: (N,4)
            rewards: (N,)
            state_s: (N,84,84,4)
        """
        
        if is_test:
            memories = self.test_memories
        else:
            memories = self.train_memories
        

            batch = random.sample(memories,batch_size)

        states,actions,rewards,state_s=map(np.array,zip(*batch))

        one_hot_actions = np.zeros((batch_size,self.action_space_size))
        one_hot_actions[np.arange(batch_size),actions] = 1

        # scale
        states = states.astype(np.float32)/255.0
        state_s = state_s.astype(np.float32)/255.0
        return states,one_hot_actions,rewards,state_s

    def _sample_memories_MultiStep(self,prediction_steps = 1,is_test = False):
        
        if is_test:
            memories = self.memories[:self.test_memory_size]
        else:
            memories = self.memories[self.test_memory_size:self.test_memory_size+self.train_memory_size]
        
        while True:
            randint = random.randint(0,len(memories)-prediction_steps)
            dones = [True if np.array_equal(memory[-1],self.zero_array) else False for memory in memories[randint:randint+prediction_steps]]
            print(dones)
            if True in dones :
                print('有内鬼',dones)
                continue
            else:
                break

        state = memories[randint][0]

        state_ = [memories[randint+step][3][:,:,-1] for step in range(prediction_steps)]

        action = [memories[randint+step][1] for step in range(prediction_steps)]

        one_hot_action = np.zeros((prediction_steps,self.action_space_size))
        one_hot_action[np.arange(prediction_steps),action] = 1

        reward = np.array([memories[randint+step][2] for step in range(prediction_steps)])

        # scale
        state = np.array(state).astype(np.float32)/255.0
        state_ = np.array(state_).astype(np.float32)/255.0
        return state,one_hot_action,reward,state_

class State_predictor:
    def __init__(self,model_path = None):
        if model_path:
            print("*** load pretrained model ***")
            with open(os.path.join(model_path,'models','model_arch.json'),'r') as f:
                self.model = model_from_json(f.read())
            self.model.load_weights(os.path.join(model_path,'models','model_weights_.h5f'))
        else:
            self.model = self._build_model()

        self.model.compile(optimizer=Adam(0.0001), loss='mse')
        
    def _build_model(self):
        frames = Input(shape = (84,84,4), name = 'frames')

        x = Conv2D(filters=64, kernel_size=6, strides=2,activation="relu")(frames)
        x = Conv2D(filters=64, kernel_size=6, strides=2,padding='same',activation="relu")(x)
        x = Conv2D(filters=64, kernel_size=6, strides=2,padding='same',activation="relu")(x)

        _,h,w,f = K.int_shape(x)


        x = Flatten()(x)

        features = Dense(1024,activation='relu')(x)

        actions = Input(shape = (4,),name = 'actions')

        features_dense = Dense(2048)(features)

        actions_dense = Dense(2048)(actions)

        features_with_action = Multiply()([features_dense,actions_dense])

        features_with_action = Dense(1024)(features_with_action)

        x = Dense(h*w*f)(features_with_action)

        x = Reshape((h,w,f))(x)

        x = Conv2DTranspose(filters=64,kernel_size=6,strides=2,padding='same',activation='relu')(x)
        x = Conv2DTranspose(filters=64,kernel_size=6,strides=2,padding='same',activation='relu')(x)
        q = Conv2DTranspose(filters=1,kernel_size=6,strides=2)(x)

        q = Reshape((84,84))(q)

        return Model(inputs=[frames,actions],outputs=q)

    def update(self,x_batch,y_batch,actions):

        loss = self.model.train_on_batch(x = {'frames':x_batch,'actions':actions},y = y_batch)

        return loss
    
    def predict(self,x,action):
        return self.model.predict_on_batch(x = {'frames':x,'actions':action})

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

# def train_world_model(model_path,epoch = 10000):
#     batch_size = 32
#     action_space_size = 4
#     ROOT_PATH = 'result_WORLD'

#     config = tf.ConfigProto(
#         gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0')
#     )

#     sess = tf.Session(config=config)
#     K.set_session(sess)

#     logger = LogWriter(ROOT_PATH,batch_size)

#     gen = Memory_generator(model_path)

    
#     print('restore memories')
#     gen.restore_memories()

#     model = state_predictor()

#     logger.set_model(model.model)
#     logger.set_loss_name([*model.model.metrics_names])


#     print('trainning world model')
#     for i in tqdm(range(epoch)):
#         states,actions,_,state_s = gen.sample_memories(batch_size)

#         loss = model.update(states,state_s[:,:,:,3],actions)

#         logger.add_loss([loss])

#     logger.save_model_arch(model)
#     logger.save_weights(model)

# def test_world_model(world_path,model_path):

#     config = tf.ConfigProto(
#         gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0')
#     )

#     sess = tf.Session(config=config)
#     K.set_session(sess)

#     from matplotlib import pyplot as plt

#     sp = state_predictor(model_path=world_path)

#     mg = Memory_generator(root_path = model_path)
#     mg.restore_memories()
    
#     while True:
#         states,actions,rewards,state_s = mg.sample_memories(batch_size = 10,test_set=True)

#         predicted_frame = sp.predict(states,actions)
        
        

#         fig = plt.figure()

#         rows,cols = 3,2

#         for i in range(1,rows+1):
#             fig.add_subplot(rows,cols,2*i-1).set_title('real')
#             plt.imshow(state_s[i-1,:,:,3],interpolation='nearest')
#             fig.add_subplot(rows,cols,2*i).set_title('predicted')
#             plt.imshow(predicted_frame[i-1,:,:],interpolation='nearest')

#         plt.show()

# def test_world_model_2(world_path,model_path,prediction_steps = 5):
#     config = tf.ConfigProto(
#         gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0')
#     )

#     sess = tf.Session(config=config)
#     K.set_session(sess)

#     sp = state_predictor(model_path=world_path)

#     mg = Memory_generator(root_path = model_path)
#     mg.restore_memories()
#     state,one_hot_action,state_ = mg.sample_memories_MultiStep(prediction_steps=prediction_steps)

#     state_=np.array(state_)
    
#     predicted_frames = []
#     current_input_state = state
#     for i in range(prediction_steps):
#         predicted_frame = sp.predict(np.expand_dims(current_input_state,axis=0),np.expand_dims(one_hot_action[i],axis=0))
#         predicted_frames.append(predicted_frame)
#         current_input_state = np.concatenate((current_input_state[:,:,1:],predicted_frame.reshape((84,84,1))),axis=2)
#         # print(current_input_state.shape)

#     predicted_frames = np.squeeze(np.array(predicted_frames))
    
#     from matplotlib import pyplot as plt

#     fig = plt.figure()

#     rows,cols = 2,prediction_steps

#     for i in range(1,cols+1):
#         fig.add_subplot(rows,cols,i).set_title('true (k+{})'.format(i))
#         plt.imshow(state_[i-1,:,:],interpolation='nearest')
#         fig.add_subplot(rows,cols,i+cols).set_title('pred (k+{})'.format(i))
#         plt.imshow(predicted_frames[i-1,:,:],interpolation='nearest')

#     plt.show()


# def distill_with_world_model(agent_model_path,world_model_path,mem_size):

#     ROOT_PATH = 'result_DT_WORLD'

#     BATCH_SIZE = 32
    
#     teacher = Teacher_world_model(agent_model_path,world_model_path,mem_size)

#     logger = LogWriter(ROOT_PATH,BATCH_SIZE)

#     student = SingleDtStudent_world(teacher,logger,'big',100000,0.0001)

#     student.distill()

#     logger.save_weights(student)
#     logger.save_model_arch(student)

if __name__ == "__main__":
    root_path = 'result/191119_214214'

    # mg = Memory_generator(root_path)
    # mg.restore_memories()
    # mg.sample_memories(32)
    # mg.sample_memories(32,test=True)
    # train_world_model('result/191031_161605',epoch=10000)

    # test_world_model_2('result_WORLD/191031_200650','result/191031_161605')

    # distill_with_world_model('result/191031_161605','result_WORLD/191031_200650',100000

    prediction_steps = 5
    mem_gen = Memory_generator(root_path)
    for i in range(100):
        state,action,reward,state_ = mem_gen._sample_memories_MultiStep(prediction_steps=prediction_steps)
     
    fig,axes = plt.subplots(1,prediction_steps+1)
    fig.set_size_inches(12,8)

    axes[0].imshow(state[:,:,-1],interpolation='nearest')
    axes[0].set_title('moto')

    for i in range(prediction_steps):
        axes[i+1].imshow(state_[i,:,:],interpolation='nearest')
        axes[i+1].set_title(str(i))


    plt.show()


    
    
    
    
    
