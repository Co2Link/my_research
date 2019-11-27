import random
import numpy as np
import os
import pickle
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_networks import ACVP
from util.decorators import timethis
from logWriter import LogWriter

torch.set_num_threads(1)


class State_predictor:
    def __init__(
        self, n_actions, memory_path="", logger=None, gpu="0", model_path=None
    ):

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
            self.device = "cuda:" + gpu
        else:
            self.device = "cpu"
        print("*** device: {} ***".format(self.device))
        self.model.to(self.device)

        if self.logger is not None:
            logger.set_loss_name([*self.model.metrics_names()])

        # Debug
        print("*** model structure ***")
        print(self.model)

    @timethis
    def _restore_memories(self, memory_path):

        with open(os.path.join(memory_path), "rb") as f:
            memories = pickle.load(f)
        self.zero_array = np.zeros(np.shape(memories[0][0]))

        # 9:1 for training and testing
        self.train_memories = memories[int(len(memories) / 10):]
        self.test_memories = memories[: int(len(memories) / 10)]

        # Mean state
        states, _, _, _ = zip(*memories)
        self.mean_state = np.mean(np.array(states), axis=(0, 3))

        print(
            "*** traning size: {},testing size: {}".format(
                len(self.train_memories), len(self.test_memories)
            )
        )

    def _sample_batch(self, batch_size=32, prediction_step=1, is_test=False):
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
            idx = random.randint(0, len(memories) - prediction_step)
            dones = [
                True if np.array_equal(memory[-1], self.zero_array) else False
                for memory in memories[idx: idx + prediction_step]
            ]
            if True in dones:
                continue
            else:
                idxes.append(idx)

        states, actions, state_s = [], [], []

        for idx in idxes:
            states.append(memories[idx][0])

            action = [memories[idx + step][1]
                      for step in range(prediction_step)]
            one_hot_action = np.zeros((prediction_step, self.n_actions))
            one_hot_action[np.arange(prediction_step), action] = 1
            actions.append(one_hot_action)

            state_ = [
                memories[idx + step][3][:, :, -1] for step in range(prediction_step)
            ]
            state_s.append(state_)

        # Data formation
        states = self._pre_process_states(np.array(states).transpose((0, 3, 1, 2)))
        state_s = self._pre_process_states(np.array(state_s))
        actions = torch.from_numpy(np.array(actions)).to(self.device).float()
        return states, actions, state_s

    def _pre_process_states(self, states):
        """
        Args:
            states: np.Array of shape (N,4,84,84)
        """
        # subtract mean_state
        states = states-self.mean_state[None, None, :, :]
        states = states.astype(np.float32) / 255.0
        states = torch.from_numpy(states).to(self.device)
        return states

    def _post_process_states(self, state):
        """
        Args:
            state: np.Array of shape (N,STEP,84,84)
        """
        return state.data.cpu().numpy()*255+self.mean_state[None, None, :, :]

    @timethis
    def save_model(self, path, info=""):
        """ save model """
        assert isinstance(info, str), "info should be str"
        suffix = "_" + info if info else ""
        path_with_file_name = "{}/{}{}.pt".format(path, "model", suffix)

        if os.path.exists(path_with_file_name):
            os.remove(path_with_file_name)

        torch.save(self.model.state_dict(), path_with_file_name)

    def predict(self, state, action):
        return self._post_process_states(self.model(state, action))

    def train_curriculum(
        self,
        curriculum_params={
            "prediction_steps": [1, 3, 5],
            "lr": [1e-4, 1e-5, 1e-5],
            "n_epoches": [int(1e5), int(2 * 1e5), int(2 * 1e5)],
        },
    ):

        for step, prediction_step in enumerate(curriculum_params["prediction_steps"]):
            print('*** curriculum leaning step: {} ***'.format(step))
            self.train(
                prediction_step=prediction_step,
                n_epoch=curriculum_params["n_epoches"][step],
                lr=curriculum_params["lr"][step],
            )

    def train(self, prediction_step=1, n_epoch=10000, lr=1e-4,loss_clip=0):

        # Standar output
        print("*** trainning params: prediction_step({}), n_epoch({}), lr({}) ***".format(
            prediction_step, n_epoch, lr))
        # Set learning rate
        for g in self.opt.param_groups:
            g['lr'] = lr

        for _ in tqdm(range(n_epoch), ascii=True):

            states, actions, state_s = self._sample_batch(
                prediction_step=prediction_step
            )

            input_states = states
            outputs = []
            for step in range(prediction_step):
                output = self.model(
                    input_states, actions[:, step, :]
                )  # shape (N,1,84,84)
                outputs.append(output)
                input_states = torch.cat(
                    (input_states[:, 1:, :, :], output), dim=1)

            outputs = torch.cat(outputs, dim=1)  # shape (N,STEP,84,84)

            loss = F.mse_loss(outputs, state_s,reduce=False)

            if loss_clip:
                loss = torch.clamp(loss,min=loss_clip)

            loss = (1 / prediction_step) * loss.sum()

            self.opt.zero_grad()

            loss.backward()

            self.opt.step()

            if self.logger is not None:
                self.logger.add_loss([loss.item()])

        if self.logger is not None:
            self.logger.save_model(self, info=str(prediction_step))

class Wrapper_sp(State_predictor):
    def __init__(self,hparams,memory_path,model_path):
        State_predictor.__init__(self,n_actions=hparams['n_action'],memory_path=memory_path,model_path=model_path)
        self.state = None # shape (1,4,84,84)

    def reset(self):
        state,_,_ = self._sample_batch(1,1)
        self.state = state
        return self._post_process_states(self.state).squeeze().transpose((1,2,0)) # shape (84,84,4)
    
    def step(self,action):
        """
        Args:
            action: int
        """
        one_hot_action = np.zeros((1,self.n_actions))
        one_hot_action[0,action] = 1
        action = torch.from_numpy(one_hot_action).to(self.device).float() # shape (1,n_action)
        next_state = self.model(self.state,action)

        self.state = torch.cat((self.state[:,1:,:,:],next_state),dim=1)
        return self._post_process_states(self.state).squeeze().transpose((1,2,0))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--loss_clip',type=float,default=0)
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--model_path',type=str,default='')
    parser.add_argument('--prediction_step',type=int,default=10)
    parser.add_argument('--model_path_1',type=str,default='')
    parser.add_argument('--model_path_2',type=str,default='')

    args = parser.parse_args()

    CURRICULUM = args.curriculum
    LOSS_CLIP = args.loss_clip
    TEST = args.test
    MODEL_PATH = args.model_path
    PREDICTION_STEP = args.prediction_step
    MODEL_PATH_1 = args.model_path_1
    MODEL_PATH_2 = args.model_path_2

    curriculum_params = {
        "prediction_steps": [1, 3, 5],
        "lr": [1e-4, 1e-5, 1e-5],
        "n_epoches": [int(1e5), int(2 * 1e5), int(2 * 1e5)],
    }

    train_params = {
        'prediction_step': 1,
        'n_epoch': 500000,
        'lr': 1e-5,
    }

    memory_path = "result/191119_214214/memories.pkl"

    args = vars(args)
    args['curriculum_params'] = str(curriculum_params)
    args['train_params'] = str(train_params)
    args['memory_path'] = memory_path

    logger = LogWriter("result_WORLD")

    logger.save_setting(args)

    if CURRICULUM:
        sp = State_predictor(4, memory_path=memory_path, logger=logger)
        sp.train_curriculum(curriculum_params=curriculum_params)
    elif TEST:
        from test.test_state_predictor import test_predict,test_predict_multi,test_predict_multi_with_models
        test_predict_multi(model_path=MODEL_PATH,prediction_step=PREDICTION_STEP)
        # test_predict_multi_with_models(model_path_1=MODEL_PATH_1,model_path_2=MODEL_PATH_2,prediction_step=PREDICTION_STEP)
    else:
        sp = State_predictor(4, memory_path=memory_path, logger=logger)
        sp.train(prediction_step=train_params['prediction_step'],
                 n_epoch=train_params['n_epoch'], lr=train_params['lr'],loss_clip=LOSS_CLIP)
