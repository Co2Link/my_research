from keras import Sequential
from keras.layers import Dense,Input,Conv2D,Flatten,Reshape,Multiply,Conv2DTranspose,Lambda
from keras.models import Model,model_from_json
from keras.optimizers import Adam
import gym
import os
import time
import numpy as np
import random
import pickle
from collections import deque,namedtuple
from keras import backend as K
import tensorflow as tf

from tqdm import tqdm

from atari_wrappers import *
from logWriter import LogWriter

Memory = namedtuple('Memory',['state','action','reward','state_'])

class Memory_generator:
    def __init__(self,root_path ,train_memory_size,test_memory_size,game_name = "BreakoutNoFrameskip-v4"):

        env = make_atari(game_name)
        self.env = wrap_deepmind(env,frame_stack=True,scale=False)

        with open(os.path.join(root_path,'model_arch.json'),'r') as f:
            self.model = model_from_json(f.read())
        
        self.model.load_weights(os.path.join(os.path.join(root_path,'model_weights.h5f')))

        self.train_memory_size = train_memory_size

        self.test_memory_size = test_memory_size

        self.root_path = root_path

        self.action_space_size = 4

        self.memories = []

    def _memory_generator(self):

        state = self.env.reset()

        while True:

            if 0.05 <=np.random.uniform(0,1):
                action = self.select_action(state)
            else:
                action = self.env.action_space.sample()
            
            state_,reward,done,_ = self.env.step(action)

            if not done:
                yield state,action,reward,state_
            else:
                # state_ = np.zeros(np.array(state_).shape)
                yield state,action,reward,state_

                state_ = self.env.reset()
            state = state_

    def generate_memories(self,store = False):

        memory_gen = self._memory_generator()

        print("*** generating memories for trainning and testing ***")
        for _ in tqdm(range(self.train_memory_size + self.test_memory_size)):
            state,action,reward,state_ = next(memory_gen)
            self.memories.append(Memory(state,action,reward,state_))
        
        if store:
            print("*** storing memories ***")
            self.store_memories()
    
    def store_memories(self):
        start_time = time.time()
        with open(os.path.join(self.root_path,'memories.pkl'),'wb') as f:
            pickle.dump(self.memories,f)
        print("*** time cost for storing memories: {} ***".format(time.time()-start_time))

    def restore_memories(self):
        start_time = time.time()
        with open(os.path.join(self.root_path,'memories.pkl'),'rb') as f:
            self.memories = pickle.load(f)
        print("*** time cost for storing memories: {} ***".format(time.time()-start_time))

    def sample_memories(self,batch_size,test = False):
        """ sample memories for trainning or testing """
        if test:
            batch = random.sample(self.memories[:self.test_memory_size],batch_size)
        else:
            batch = random.sample(self.memories[self.test_memory_size:self.train_memory_size],batch_size)

        states,actions,rewards,state_s=map(np.array,zip(*batch))

        one_hot_actions = np.zeros((batch_size,self.action_space_size))
        one_hot_actions[np.arange(batch_size),actions] = 1

        # scale
        states = np.array(states).astype(np.float32)/255.0
        state_s = np.array(state_s).astype(np.float32)/255.0
        
        return states,one_hot_actions,rewards,state_s

    def test(self):
        states,actions,rewards,state_s = self.sample_memories(batch_size = 32)

        empty_state = np.zeros(state_s[0].shape)

        count_1 = 0

        for i in range(32):
            if np.array_equal(state_s[i],empty_state):
                count_1+=1
                continue
            else:
                frame_1 = states[i,:,:,3]
                frame_2 = state_s[i,:,:,2]
                assert np.array_equal(frame_1,frame_2),'error'
        print(states.shape)
        print(actions.shape)
        print(count_1)

    def play(self):
        state = self.env.reset()

        while True:
            self.env.render()

            if 0.05 <= np.random.uniform(0,1):
                action = self.select_action(state)
            else:
                action = self.env.action_space.sample()

            state_, _, done, _ = self.env.step(action)

            if done:
                state_ = self.env.reset()
            
            state = state_

    def select_action(self, state):
        state = self._LazyFrame2array(state)
        output = self.model.predict_on_batch(
            np.expand_dims(state, axis=0)).ravel()
        return np.argmax(output)

    def _LazyFrame2array(self, lazyframe):
        return np.array(lazyframe)

class state_predictor:
    def __init__(self,model_path = None):
        if model_path:
            print("*** load pretrained model ***")
            with open(os.path.join(model_path,'model_arch.json'),'r') as f:
                self.model = model_from_json(f.read())
            self.model.load_weights(os.path.join(model_path,'model_weights_.h5f'))
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

def train_world_model(restore_memories=False):
    batch_size = 32
    action_space_size = 4
    epoch = 10000
    ROOT_PATH = 'result_WORLD'

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0')
    )

    sess = tf.Session(config=config)
    K.set_session(sess)

    logger = LogWriter(ROOT_PATH,batch_size)

    gen = Memory_generator(root_path='model',memory_size=100000)

    
    if restore_memories:
        print('restore memories')
        gen.restore_memories()
    else:
        print('generating memories')
        gen.generate_memories(store=True)

    model = state_predictor()

    logger.set_model(model.model)
    logger.set_loss_name([*model.model.metrics_names])


    print('trainning world model')
    for i in tqdm(range(epoch)):
        states,actions,rewards,state_s = gen.sample_trainning_memories(batch_size)

        loss = model.update(states,state_s[:,:,:,3],actions)

        logger.add_loss([loss])

    logger.save_model_arch(model)
    logger.save_weights(model)

def test_world_model():

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list='0')
    )

    sess = tf.Session(config=config)
    K.set_session(sess)

    sp = state_predictor(model_path='result_WORLD/191029_234644/models')

    mg = Memory_generator(root_path = 'model',memory_size = 1000)
    mg.restore_memories()
    states,actions,rewards,state_s = mg.sample_trainning_memories(10)

    predicted_frame = sp.predict(states,actions)
    
    print(predicted_frame.shape)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10,10))

    rows,cols = 3,2

    for i in range(1,rows+1):
        fig.add_subplot(rows,cols,2*i-1).set_title('real')
        plt.imshow(state_s[i-1,:,:,3],interpolation='nearest')
        fig.add_subplot(rows,cols,2*i).set_title('predicted')
        plt.imshow(predicted_frame[i-1,:,:],interpolation='nearest')

    plt.show()


    


if __name__ == "__main__":
    # train_world_model(restore_memories=True)
    # test_world_model()
    mg = Memory_generator(root_path = 'model',train_memory_size = 1000, test_memory_size = 500)
    mg.generate_memories(store = True)
    mg.test()







        

        
        
