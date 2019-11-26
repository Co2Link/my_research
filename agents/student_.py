import torch.optim as optim
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm

from agents.ddqn import DDQN
from rl_networks import Nature_CNN

def kullback_leibler_divergence(output, target):
    tau = 0.1

    target = F.softmax(target/tau,dim=1)
    output = F.softmax(output,dim=1)

    return (target*torch.log(target/output)).sum()

class SingleDtStudent:
    def __init__(self, logger, hparams, gpu='0'):
                # Hpyerparameters
        self.hparams = hparams

        self.logger = logger

        assert hparams['net_size'] in ['big', 'small', 'normal',
                                       'super'], "net_size must be one of ['big','small','normal']"

        self.model = Nature_CNN(
            self.hparams['n_actions'], hparams['net_size'])

        self.opt = optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

        if logger is not None:
            logger.set_loss_name([*self.model.metrics_names()])

        # Check gpu
        if torch.cuda.is_available:
            self.device = 'cuda:' + gpu
        else:
            self.device = 'cpu'

        self.model.to(self.device)

    def _input_to_device(self,x):
        x = x.astype(np.float32)/255.0
        x = torch.Tensor(x.transpose((0, 3, 1, 2))).to(self.device)
        return x

    def distill(self, teacher):

        for _ in tqdm(range(self.hparams['epoch'])):
            
            teacher.add_memories(self.hparams['add_mem_num'])

            for _ in range(self.hparams['n_update']):
                s_b,o_b = teacher.sample_memories()

                inputs = self._input_to_device(s_b)
                targets = torch.from_numpy(o_b).to(device=self.device)

                outputs = self.model(inputs)

                loss = kullback_leibler_divergence(outputs,targets)

                self.opt.zero_grad()
                
                loss.backward()

                self.opt.step()

                if self.logger is not None:
                    self.logger.add_loss([loss.item()])

    def save_model(self, path, info=''):
        """ save model """
        assert isinstance(info, str), 'info should be str'
        suffix = '_'+info if info else ''
        path_with_file_name = "{}/{}{}.pt".format(path, 'model', suffix)

        if os.path.exists(path_with_file_name):
            os.remove(path_with_file_name)

        torch.save(self.model.state_dict(), path_with_file_name)
