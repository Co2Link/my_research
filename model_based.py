import random
import numpy as np
import os
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_networks import ACVP
from util.decorators import timethis
from logWriter import LogWriter


class State_predictor:
    def __init__(self, n_actions, prediction_steps = 1, memory_path = '',logger = None, gpu='0', model_path=None):

        self.prediction_steps = prediction_steps

        self.n_actions = n_actions

        self.logger = logger

        # Create or load a model
        self.model = ACVP(n_actions)
        if model_path:
            print("*** loaded model: {} ***".format(os.path.basename(model_path)))
            self.model.load_state_dict(torch.load(model_path))

        self.opt = torch.optim.Adam(self.model.parameters(), 1e-4)

        # Load memory
        if memory_path:
            self._restore_memories(memory_path)

        # Check gpu
        if torch.cuda.is_available:
            self.device = 'cuda:' + gpu
        else:
            self.device = 'cpu'
        print('*** device: {} ***'.format(self.device))
        self.model.to(self.device)

        if self.logger is not None:
            logger.set_loss_name([*self.model.metrics_names()])

        # Debug
        print('*** model structure ***')
        print(self.model)

    @timethis
    def _restore_memories(self, memory_path):

        with open(os.path.join(memory_path), 'rb') as f:
            memories = pickle.load(f)
        self.zero_array = np.zeros(np.shape(memories[0][0]))

        # 9:1 for training and testing
        self.train_memories = memories[int(len(memories)/10):]
        self.test_memories = memories[:int(len(memories)/10)]
        print('*** traning size: {},testing size: {}'.format(
            len(self.train_memories), len(self.test_memories)))

    def _sample_batch(self, batch_size, is_test=False):
        """
        Output:
            states: batch of current state (N,4,84,84)
            actions: batch of action (N,STEP,N_ACTION)
            state_s: batch of next state (N,STEP,84,84)
        """

        # Test or Train
        if is_test:
            memories = self.test_memories
        else:
            memories = self.train_memories

        # Indexes for sample that dont contain zero_array
        idxes = []
        while len(idxes) < batch_size:
            idx = random.randint(0, len(memories)-self.prediction_steps)
            dones = [True if np.array_equal(
                memory[-1], self.zero_array) else False for memory in memories[idx:idx+self.prediction_steps]]
            if True in dones:
                continue
            else:
                idxes.append(idx)

        states, actions, state_s = [], [], []

        for idx in idxes:
            states.append(memories[idx][0])

            action = [memories[idx+step][1]
                      for step in range(self.prediction_steps)]
            one_hot_action = np.zeros((self.prediction_steps, self.n_actions))
            one_hot_action[np.arange(self.prediction_steps), action] = 1
            actions.append(one_hot_action)

            state_ = [memories[idx+step][3][:, :, -1]
                      for step in range(self.prediction_steps)]
            state_s.append(state_)

        # Data formation
        states = np.array(states).astype(np.float32)/255.0
        states = torch.Tensor(states.transpose((0, 3, 1, 2))).to(
            self.device)  # (N,84,84,4) => (N,4,84,84)
        actions = torch.from_numpy(np.array(actions)).to(self.device).float()
        state_s = np.array(state_s).astype(np.float32)/255.0
        state_s = torch.from_numpy(state_s).to(self.device)
        return states, actions, state_s

    def save_model(self, path, info=''):
        """ save model """
        assert isinstance(info, str), 'info should be str'
        suffix = '_'+info if info else ''
        path_with_file_name = "{}/{}{}.pt".format(path, 'model', suffix)

        if os.path.exists(path_with_file_name):
            os.remove(path_with_file_name)

        torch.save(self.model.state_dict(), path_with_file_name)

    def predict(self, state, action):
        return self.model(state, action)

    def train(self, batch_size=32, n_epoch=10000):

        for _ in tqdm(range(n_epoch)):

            states, actions, state_s = self._sample_batch(batch_size)

            input_states = states
            outputs = []
            for step in range(self.prediction_steps):
                # shape (N,1,84,84)
                output = self.model(input_states, actions[:, step, :])
                print(output.size())
                outputs.append(output)
                input_states = torch.cat(
                    (input_states[:, :3, :, :], output), dim=1)

            outputs = torch.cat(outputs, dim=1)  # shape (N,STEP,84,84)

            loss = (1/self.prediction_steps)*F.mse_loss(outputs, state_s)

            self.opt.zero_grad()

            loss.backward()

            self.opt.step()

            if self.logger is not None:
                self.logger.add_loss([loss.item()])
        
        if self.logger is not None:
            self.logger.save_model(self)

def test_sample():
    import matplotlib.pyplot as plt
    memory_path = 'result/191119_214214/memories.pkl'

    sp = State_predictor(4, 4, memory_path)
    states, actions, state_s = sp._sample_batch(32)
    print(states.shape, actions.shape, state_s.shape)

    fig, axes = plt.subplots(1, 5)

    axes[0].imshow(states[0, :, :, -1], interpolation='nearest')
    axes[0].set_title('moto')
    for i in range(4):
        axes[1+i].imshow(state_s[0, i, :, :], interpolation='nearest')
        axes[i+1].set_title(str(i+1))
    plt.show()

def test_predict():
    sp = State_predictor(4,memory_path = memory_path,model_path = model_path,)

    states,actions,state_s = sp._sample_batch(32)
    print(actions.size(),state_s.size())
    output = sp.predict(states,actions[:,0,:])

    output = output.data.cpu().numpy()
    state_s = state_s.data.cpu().numpy()
    states = states.data.cpu().numpy()


    for b in range(32):
        fig,axes = plt.subplots(1,5)

        for i in range(3):
            axes[i].imshow(states[b,i+1,:,:],interpolation='nearest')
            axes[i].set_title(str(i))

        axes[3].imshow(output[b,0,:,:],interpolation='nearest')
        axes[3].set_title('3')
        axes[4].imshow(state_s[b,0,:,:],interpolation='nearest')
        axes[4].set_title('4')

        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    memory_path = 'result/191119_214214/memories.pkl'
    model_path = 'result_world/191122_164422/models/model.pt'

    # train
    logger = LogWriter('result_world')

    sp = State_predictor(4,1,memory_path= memory_path,logger=logger)
    sp.train(n_epoch=100000)


