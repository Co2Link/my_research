import torch
import numpy as np
import os
from collections import deque

from agents.ddqn import DDQN
from atari_wrappers import make_atari,wrap_deepmind

class Teacher(DDQN):

    def __init__(self, load_model_path, hparams,env):
        DDQN.__init__(self,logger=None, load_model_path=load_model_path,
                      hparams=hparams, inference=True)

        self.mem_gen = self._memory_generator(env)

        self.s_m=deque(maxlen=hparams['mem_size'])
        self.o_m=deque(maxlen=hparams['mem_size'])
        
    def _select_action(self,state):
        state = self.input_to_device(np.expand_dims(np.array(state), axis=0))
        output = self.model(state)
        action = torch.argmax(output).item()
        return action,np.squeeze(output.data.cpu().numpy())

    def _memory_generator(self,env):

        state = env.reset()

        while True:

            if self.hparams['eps'] <= np.random.uniform(0, 1):
                action, output = self._select_action(state)
                # generate memory
                yield state, output
            else:
                action = env.action_space.sample()

            # step
            state_, _, done, _ = env.step(action)

            # reset environment
            if done:
                state_ = env.reset()

            state = state_

    def add_memories(self,size):
        for _ in range(size):
            state,output = next(self.mem_gen)
            self.s_m.append(state)
            self.o_m.append(output)
        print("*** add {} memories,memory size: {} ***".format(size, len(self.s_m)))
    
    def sample_memories(self,batch_size=32):

        index = np.random.choice(len(self.s_m), batch_size)
        index = list(index)

        s_batch = [self.s_m[ind] for ind in index]
        o_batch = [self.o_m[ind] for ind in index]

        return np.array(s_batch), np.array(o_batch)

    
class Teacher_world_model(DDQN):
    def __init__(self, load_model_path, hparams,state_predictor):
        DDQN.__init__(self,logger=None, load_model_path=load_model_path,
                      hparams=hparams, inference=True)

        self.mem_gen = self._memory_generator()

        self.sp = state_predictor

        self.s_m=deque(maxlen=hparams['mem_size'])
        self.o_m=deque(maxlen=hparams['mem_size'])
        
    def _memory_generator(self):
        state,action,_ = self.sp._sample_batch(self,1,1)
        input_state = state

        self.model(input_state)
        next_state = self.sp.model(state,action[:,0,:])

        input_state = torch.cat((input_state[:,1:,:,:],next_state),dim=1)