from keras import Sequential
from keras.layers import Dense,Input,Conv2D,Flatten,Reshape,Multiply
from keras.models import Model,model_from_json
import gym
import os
import numpy as np
from collections import deque,namedtuple
from keras import backend as K

from tqdm import tqdm

from atari_wrappers import *

class Memory_generator:
    def __init__(self,model_path,memory_size,game_name = "BreakoutNoFrameskip-v4"):

        env = make_atari(game_name)
        self.env = wrap_deepmind(env,frame_stack=True,scale=True)

        with open(os.path.join(model_path,'model_arch.json'),'r') as f:
            self.model = model_from_json(f.read())
        
        self.model.load_weights(os.path.join(os.path.join(model_path,'model_weights.h5f')))

        self._memory = deque(maxlen=memory_size)
        self.memory = deque(maxlen=memory_size)

        self.memory_size = memory_size

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
                state_ = np.zeros(np.array(state_).shape)
                yield state,action,reward,state_

                state_ = self.env.reset()
            state = state_

    def _generate_memory(self):

        Memory = namedtuple('Memory',['state','action','reward','state_'])

        memory_gen = self._memory_generator()

        for _ in tqdm(range(self.memory_size)):
            state,action,reward,state_ = next(memory_gen)
            self._memory.append(Memory(state,action,reward,state_))

    def test(self):
        self._generate_memory()
        print(len(self._memory))
        states,actions,rewards,state_s=map(np.array,zip(*self._memory))

        empty_state = np.zeros(state_s[0].shape)

        count_1 = 0

        for i in range(self.memory_size):
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
    def __init__(self):
        self.model = _build_model()
        self.model.compile(optimizer=Adam(0.0001), loss='mse')



    def _build_model(self):
        frames = Input(shape = (84,84,4), name = 'frames')

        x = Conv2D(filters=64, kernel_size=6, strides=2,padding='same',activation="relu", name=(name + "_conv2D_1"))(frames)
        x = Conv2D(filters=64, kernel_size=6, strides=2,padding='same',activation="relu", name=(name + "_conv2D_2"))(x)
        x = Conv2D(filters=64, kernel_size=6, strides=2,padding='same',activation="relu", name=(name + "_conv2D_3"))(x)

        print("***",K.int_shape(x))
        h,w,f = K.int_shape(x)

        print("****",h,w,f)

        x = Flatten()(x)

        features = Dense(1024,activation='relu',name=(name+"_dense_1"))(x)

        actions = Input(shape = (4,),name = 'actions')

        features_dense = Dense(2048,name=(name+'_features_dense'))(features)

        actions_dense = Dense(2048,name=(name+'_actions_dense'))(actions)

        features_with_action = Multiply()([features_dense,actions_dense])

        features_with_action = Dense(1024)(features_with_action)

        x = Dense(h*w*f)(features_with_action)

        x = Reshape((h,w,f))(x)

        x = Conv2DTranspose(filters=64,kernel_size=6,strides=2,padding='same',activation='relu',name = (name+'_deconv2d_1'))(x)
        x = Conv2DTranspose(filters=64,kernel_size=6,strides=2,padding='same',activation='relu',name = (name+'_deconv2d_2'))(x)
        q = Conv2DTranspose(filters=1,kernel_size=6,strides=2,padding='same',name = (name+'_deconv2d_3'))(x)

        return Model(inputs=[frames,actions],outputs=q)

    def update(self,x_batch,y_batch,actions):

        loss = self.model.train_on_batch(x = {'frames':x_batch,'actions':actions},y = y_batch)
    
    def predict(self,x,action):
        return self.model.predict_on_batch(np.expand_dims(state,axis)).ravel()

if __name__ == "__main__":
    gen = Memory_generator(model_path='model',memory_size=10000)
    gen.test()
        

        
        

        