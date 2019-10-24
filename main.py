import argparse
from gym import wrappers
from agents.ddqn import DDQN
from runners.normal_runner import Normal_runner
from logWriter import LogWriter
from keras import backend as K
import tensorflow as tf
import time
from atari_wrappers import *


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

    ddqn_agent = DDQN(
        env, LEARNING_RATE, GAMMA, logger, MAX_MEM_LEN, BATCH_SIZE, SCALE, IS_SMALL, IS_LOAD_MODEL
    )

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
        logger.save_weights(ddqn_agent, -1)
        # Record the total used time
        logger.log_total_time_cost()


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

    ddqn_agent = DDQN(env, LEARNING_RATE, GAMMA, logger, MAX_MEM_LEN, BATCH_SIZE, SCALE, IS_SMALL, IS_LOAD_MODEL)

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
        logger.save_weights(ddqn_agent, -1)
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
    parser.add_argument("--no_scale", action="store_false")
    parser.add_argument("--is_small", action="store_true")
    parser.add_argument("--is_load_model", action="store_true")
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
    SCALE = args.no_scale
    ROOT_PATH = args.root
    IS_SMALL = args.is_small
    IS_LOAD_MODEL = args.is_load_model

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=args.gpu)
    )
    sess = tf.Session(config=config)
    K.set_session(sess)

    logger = LogWriter(ROOT_PATH, BATCH_SIZE)
    logger.save_setting(args)

    if args.test:
        test(logger)
    else:
        ddqn_main(logger)
