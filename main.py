import argparse
from gym import wrappers
from agents.ddqn import DDQN
from agents.base import MemoryStorer
from runners.normal_runner import Normal_runner
from logWriter import LogWriter
import time
import torch
from atari_wrappers import *

torch.set_num_threads(1)


def ddqn_main(logger):
    # make environment
    env = make_atari(GAME)

    if logger is not None:
        env = wrappers.Monitor(
            env,
            logger.get_movie_pass(),
            video_callable=(lambda ep: ep % 100 == 0),
            force=True,
        )

    env = wrap_deepmind(env, frame_stack=True, scale=SCALE)

    runner = Normal_runner(
        EPS_START, EPS_END, EPS_STEP, logger, RENDER, SAVE_MODEL_INTERVAL
    )

    hparams = {'lr': LEARNING_RATE, 'gamma': GAMMA, 'memory_size': MAX_MEM_LEN, 'batch_size': BATCH_SIZE,
               'scale': SCALE, 'net_size': NET_SIZE, 'state_shape': env.observation_space.shape, 'n_actions': env.action_space.n}

    ddqn_agent = DDQN(logger, LOAD_MODEL_PATH, hparams)

    memory_storer = MemoryStorer(
        MEMORY_SOTRATION_SIZE) if MEMORY_SOTRATION_SIZE else None

    runner.train(
        ddqn_agent,
        memory_storer,
        env,
        MAX_ITERATION,
        BATCH_SIZE,
        WARMUP,
        TARGET_UPDATE,
    )

    if logger is not None:
        # Save the final model
        logger.save_model(ddqn_agent, '-1')
        # Record the total used time
        logger.log_total_time_cost()
        # save the memories
        logger.store_memories(memory_storer)


def test(logger):
    MAX_ITERATION = 5000
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_STEP = 1000
    MAX_MEM_LEN = 1000
    WARMUP = 1000
    TARGET_UPDATE = 1000
    SAVE_MODEL_INTERVAL = 100
    RENDER = False
    GAME = "BreakoutNoFrameskip-v4"
    SCALE = True

    # make environment
    env = make_atari(GAME)

    if logger is not None:
        env = wrappers.Monitor(
            env,
            logger.get_movie_pass(),
            video_callable=(lambda ep: ep % 100 == 0),
            force=True,
        )

    env = wrap_deepmind(env, frame_stack=True, scale=SCALE)

    runner = Normal_runner(
        EPS_START, EPS_END, EPS_STEP, logger, RENDER, SAVE_MODEL_INTERVAL
    )

    ddqn_agent = DDQN(env, LEARNING_RATE, GAMMA, logger, MAX_MEM_LEN,
                      BATCH_SIZE, SCALE, NET_SIZE, LOAD_MODEL_PATH, MEMORY_SOTRATION_SIZE)

    runner.train(
        ddqn_agent,
        env,
        MAX_ITERATION,
        BATCH_SIZE,
        warmup=WARMUP,
        target_update_interval=TARGET_UPDATE,
    )

    if logger is not None:
        # Save the final model
        logger.save_weights(ddqn_agent, '-1')
        # Record the total used time
        logger.log_total_time_cost()


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-iter", "--max_iteration", type=int, default=int(1e6))
    parser.add_argument("-b", "--batchsize", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("-e_start", "--eps_start", type=float, default=1.0)
    parser.add_argument("-e_end", "--eps_end", type=float, default=0.1)
    parser.add_argument("-e_step", "--eps_step", type=int, default=int(1e5))
    parser.add_argument("-m_len", "--max_mem_len", type=int, default=int(1e4))
    parser.add_argument("--warmup", type=int, default=int(1e4))
    parser.add_argument("--target_update", type=int, default=1000)
    parser.add_argument("-s", "--save_model_interval", type=int, default=100)
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("--root", type=str, default="./result")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--game", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--store_memory", action="store_true")
    parser.add_argument("--net_size", type=str, default="normal")
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--memory_storation_size", type=int, default=100000)

    args = parser.parse_args()

    MAX_ITERATION = args.max_iteration
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batchsize
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_STEP = args.eps_step
    MAX_MEM_LEN = args.max_mem_len
    WARMUP = args.warmup
    TARGET_UPDATE = args.target_update
    SAVE_MODEL_INTERVAL = args.save_model_interval
    RENDER = args.render
    GAME = args.game
    SCALE = args.scale
    ROOT_PATH = args.root
    NET_SIZE = args.net_size
    LOAD_MODEL_PATH = args.load_model_path
    MEMORY_SOTRATION_SIZE = args.memory_storation_size

    assert MEMORY_SOTRATION_SIZE < MAX_ITERATION,'MEMORY_SOTRATION_SIZE < MAX_ITERATION'

    logger = LogWriter(ROOT_PATH, BATCH_SIZE)
    logger.save_setting(args)

    if args.test:
        test(logger)
    else:
        ddqn_main(logger)
