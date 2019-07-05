import numpy as np

from runners.rl_runner import RL_runner

import time

class Normal_runner(RL_runner):
    def __init__(self,memory_size,eps_start=1.0,eps_end=0.01,eps_step=1e5,logger=None):
        super().__init__(logger)

        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_step=float(eps_step)

    def traj_generator(self,agent,env):
        """ generate trajectory """
        state=env.reset()

        total_step=0

        epi_step=0

        episode_reward=0

        while True:

            env.render()

            # eps-greedy
            fraction=min(step,self.eps_step)/self.eps_step
            if self.eps_start+(self.eps_end-self.eps_start)*fraction <=np.random.uniform(0,1):
                action=agent.select_action(state)
            else:
                action=env.observation_space.sample()

            # step
            state_,reward,done,_=env.step(action)

            # log
            episode_reward+=reward
            epi_step+=1
            total_step+=1

            if done == False:
                yield state,action,reward,state_
            else:
                state_=np.zeros((state_.shape))

                yield state_,action,reward,state_

                self.epi_count+=1

                epi_step=0

                episode_reward=0

                state_=env.reset()

            state=state_

    def test(self,agent,env,iter=1):
        pass

    def train(self,agent,env,max_iter,batch_size,time_step=0,warmup=0,target_update_interval=0):
        pass

def DEBUG():
    pass


if __name__ == '__main__':
    DEBUG()
