import numpy as np

from runners.rl_runner import RL_runner

import time


class Normal_runner(RL_runner):
    def __init__(self, eps_start=1.0, eps_end=0.01, eps_step=1e5, logger=None, render=False, save_model_interval=100):
        super().__init__(logger)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_step = float(eps_step)
        self.render = render
        self.save_model_interval = save_model_interval


    def traj_generator(self, agent, env):
        """ generate trajectory """
        state = env.reset()

        total_step = 0.0

        epi_step = 0

        episode_reward = 0

        while True:

            if self.render:
                env.render()

            # eps-greedy
            fraction = min(total_step, self.eps_step) / self.eps_step
            if self.eps_start + (self.eps_end - self.eps_start) * fraction <= np.random.uniform(0, 1):
                action = agent.select_action(state)
            else:
                action = env.action_space.sample()

            # step
            state_, reward, done, _ = env.step(action)

            # log
            episode_reward += reward
            epi_step += 1
            total_step += 1

            if self.logger is not None:
                self.logger.count_iteration()

            if not done:
                yield state, action, reward, state_
            else:
                # state_ = np.zeros(np.array(state_).shape)

                yield state, action, reward, state_

                # log
                if self.logger is not None:
                    if self.episode % self.save_model_interval == 0:
                        self.logger.save_weights(agent, self.episode)

                    self.logger.add_reward(self.episode, episode_reward, {"steps": epi_step, "epsilon": (
                        self.eps_start + (self.eps_end - self.eps_start) * fraction)})
                else:
                    print("{} : {}, steps : {}, epsilon : {}".format(self.episode, episode_reward, epi_step,
                                                                     self.eps_start + (
                                                                         self.eps_end - self.eps_start) * fraction))

                self.episode += 1
                epi_step = 0
                episode_reward = 0

                # reset environment
                state_ = env.reset()

            state = state_

    def test(self, agent, env, iter=1):
        pass

    def train(self, agent, env, max_iter, batch_size, warmup=0, target_update_interval=0):
        """ train the agent """

        self.logger.save_model_arch(agent)

        traj_gen = self.traj_generator(agent, env)

        start_time = time.time()
        for i in range(max_iter):

            s, a, r, ns = next(traj_gen)

            agent.memorize(s, a, r, ns)

            if i > warmup and i % 4 == 0:

                agent.learn()

                if i % target_update_interval == 0:
                    agent.target_update()

            if i % 1000 == 0 and i != 0:
                # check time cost
                print("*** 1e3 iteration cost: {:.5f}, memory size: {:,}, *** ,Progress: {:.2f}% ***".format(
                    time.time() - start_time, agent.memory_size(), i * 100 / max_iter))
                start_time = time.time()


def DEBUG():
    pass


if __name__ == '__main__':
    DEBUG()
