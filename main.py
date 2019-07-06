import argparse
from gym import wrappers
from agents.ddqn import DDQN
from runners.normal_runner import Normal_runner
from logWriter import LogWriter
from keras import backend as K
import tensorflow as tf

from atari_wrappers import *

# 学習用定数
MAX_ITERATION = 1000000
LEARNUNG_RATE = 0.001
BATCH_SIZE = 32
EPISODE = None
GAMMA = 0.99

EPS_START = 1.0
EPS_END = 0.001
EPS_STEP = 1e5

MAX_MEM_LEN = 10000
WARMUP = 10000
TARGET_UPDATE = 1000

RENDER = False
LOG=False

SAVE_MODEL_INTERVAL = 100

ROOT_PATH = "./root"


def ddqn_main():
    # make environment
    env = make_atari("BreakoutNoFrameskip-v4")

    # TensorBoard
    if LOG:

        logger = LogWriter(ROOT_PATH, BATCH_SIZE)

        # save movies
        env = wrappers.Monitor(env, logger.get_movie_pass(), video_callable=(lambda ep: ep % 100 == 0), force=True)

    else:
        logger=None

    env = wrap_deepmind(env, frame_stack=True, scale=True)

    runner = Normal_runner(EPS_START, EPS_END, EPS_STEP, logger, RENDER)

    ddqn_agent = DDQN(env, LEARNUNG_RATE, GAMMA, logger, 1e4, 32)

    runner.train(ddqn_agent, env, MAX_ITERATION, BATCH_SIZE, warmup=WARMUP, target_update_interval=TARGET_UPDATE)

    ddqn_agent.save_model(-1, "model.hdf5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', '--max_iteration', type=int, default=1000000)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('-e_start', '--eps_start', type=float, default=1.0)
    parser.add_argument('-e_end', '--eps_end', type=float, default=0.01)
    parser.add_argument('-e_step', '--eps_step', type=float, default=1e5)
    parser.add_argument('-m_len', '--max_mem_len', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('-s', '--save_model_interval', type=int, default=100)
    parser.add_argument('-r', '--root_path', type=str, default="./root")
    parser.add_argument('-a_type', '--augment_type', type=str, default="None")
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--log', action="store_true")
    args = parser.parse_args()

    MAX_ITERATION = args.max_iteration
    LEARNUNG_RATE = args.learning_rate
    BATCH_SIZE = args.batchsize
    GAMMA = args.gamma
    EPS_START = args.eps_start
    EPS_END = args.eps_end
    EPS_STEP = args.eps_step
    MAX_MEM_LEN = args.max_mem_len
    WARMUP = args.warmup
    TARGET_UPDATE = args.target_update
    SAVE_MODEL_INTERVAL = args.save_model_interval
    ROOT_PATH = args.root_path
    RENDER=args.render
    LOG=args.log

    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, visible_device_list=args.gpu))
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    ddqn_main()