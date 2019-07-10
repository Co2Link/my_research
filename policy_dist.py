from keras.models import load_model
import numpy as np
import os
import gym
from atari_wrappers import wrap_deepmind,make_atari
model_path='./teacher'

class Teacher():
    def __init__(self,model,game_name,epsilon):

        self.model=model

        # make environment
        env=make_atari(game_name)
        env=wrap_deepmind(env,frame_stack=True,scale=True)
        self.env=env

        self.epsilon=epsilon

    def select_action(self,state):
        state=self.LazyFrame2array(state)
        action=self.model.predict_on_batch(np.array([state]))
        return np.argmax(a[0])

    def memory_generator(self):
        state=self.env.reset()

        while True:

            if self.epsilon<=np.random.uniform(0,1) :
                action=self.select_action(state)
            else:
                action=self.env.action_space.sample()

            # step
            state_,reward,done,_=self.env.step(action)

            # generate memory
            if not done:
                yield state,action,reward,state_
            else:
                state_=np.zeros(np.array(state_).shape)

                yield state,action,reward,state_


    def LazyFrame2array(self,lazyframe):
        return np.array(lazyframe)

def DEBUG():
    pass

if __name__ == '__main__':
    DEBUG()