import random
import numpy as np
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_networks import ACVP
from util.decorators import timethis


class State_predictor:
    def __init__(self,n_actions,prediction_steps,memory_path,gpu = '0',model_path = None):

        self.prediction_steps = prediction_steps

        self.n_actions = n_actions

        # TODO load model

        self.model = ACVP(n_actions)
        self.criterion=nn.MSELoss()
        self.opt=torch.optim.Adam(self.model.parameters(), 1e-4)

        # Load memory
        self._restore_memories(memory_path)

        # Check gpu
        if torch.cuda.is_available:
            self.device = 'cuda:' + gpu
        else:
            self.device = 'cpu'
        print('*** device: {} ***'.format(self.device))
        self.model.to(self.device)

        # Debug
        print('*** model structure ***')
        print(self.model)

    @timethis
    def _restore_memories(self,memory_path):

        with open(os.path.join(memory_path),'rb') as f:
            memories = pickle.load(f)
        self.zero_array = np.zeros(np.shape(memories[0][0]))
        
        # 9:1 for training and testing
        self.train_memories = memories[int(len(memories)/10):]
        self.test_memories = memories[:int(len(memories)/10)]
        print('*** traning size: {},testing size: {}'.format(len(self.train_memories),len(self.test_memories)))

    def _sample_batch(self,batch_size,is_test = False):
        """
        Out shape:
            states: (N,84,84,4)
            actions: (N,STEP,N_ACTION)
            state_s: (N,STEP,84,84)
        """

        # Test or Train
        if is_test:
            memories = self.test_memories
        else:
            memories = self.train_memories

        # Indexes for sample that dont contain zero_array
        idxes = []
        while len(idxes) < batch_size:
            idx = random.randint(0,len(memories)-self.prediction_steps)
            dones = [True if np.array_equal(memory[-1],self.zero_array) else False for memory in memories[idx:idx+self.prediction_steps]]
            if True in dones :
                continue
            else:
                idxes.append(idx)
        
        states,actions,state_s = [],[],[]
        
        for idx in idxes:
            states.append(memories[idx][0])

            action = [memories[idx+step][1] for step in range(self.prediction_steps)]
            one_hot_action = np.zeros((self.prediction_steps,self.n_actions))
            one_hot_action[np.arange(self.prediction_steps),action] = 1
            actions.append(one_hot_action)

            state_ = [memories[idx+step][3][:,:,-1] for step in range(self.prediction_steps)]
            state_s.append(state_)

        return np.array(states),np.array(actions),np.array(state_s)
        

    
    def predict(self,state,action):
        return self.model(state,action)

    def update(self,state_batches,action_batches,next_state_batches):
        
        y = self.model(state_batches,action_batches)
        loss = F.mse_loss(y,next_state_batches)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    memory_path = 'result/191119_214214/memories.pkl'

    sp = State_predictor(4,4,memory_path)
    states,actions,state_s = sp._sample_batch(32)
    print(states.shape,actions.shape,state_s.shape)

    fig,axes = plt.subplots(1,5)

    axes[0].imshow(states[0,:,:,-1],interpolation='nearest')
    axes[0].set_title('moto')
    for i in range(4):
        axes[1+i].imshow(state_s[0,i,:,:],interpolation='nearest')
        axes[i+1].set_title(str(i+1))
    plt.show()