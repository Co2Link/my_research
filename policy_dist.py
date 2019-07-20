import argparse
from gym import wrappers
from agents.ddqn import DDQN
from runners.normal_runner import Normal_runner
from logWriter import LogWriter
from keras import backend as K
import tensorflow as tf
import time
from atari_wrappers import *

from agents.teacher import Teacher
from agents.student import SingleDtStudent


def SingleDistillation_main(logger):

    env=make_atari(GAME)
    env=wrap_deepmind(env,frame_stack=True,scale=True)

    teacher=Teacher(MODEL_PATH, env, EPSILON, MEM_SIZE)

    student=SingleDtStudent(env,LEARNING_RATE,logger,BATCH_SIZE,EPSILON,teacher,ADD_MEM_NUM,UPDATE_NUM,EPOCH)

    student.distill()

    if logger is not None:
        logger.save_model(student,-1)
        logger.log_total_time_cost()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-g', '--game', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.05)
    parser.add_argument('--add_mem_num', type=int, default=int(5e3))
    parser.add_argument('--update_num', type=int, default=int(1e3))
    parser.add_argument('-ep', '--epoch', type=int, default=20)
    parser.add_argument('--mem_size',type=int,default=int(5e4))
    parser.add_argument('-r', '--root_path', type=str, default='./result_DT')
    parser.add_argument('--model_path', type=str, default='./teacher/breakout.h5f')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    LEARNING_RATE = args.learning_rate
    GAME = args.game
    BATCH_SIZE = args.batch_size
    EPSILON = args.epsilon
    ADD_MEM_NUM = args.add_mem_num
    UPDATE_NUM = args.update_num
    EPOCH = args.epoch
    MEM_SIZE = args.mem_size
    ROOT_PATH = args.root_path
    MODEL_PATH = args.model_path
    LOG = args.log

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    K.set_session(sess)

    if LOG:
        logger = LogWriter(ROOT_PATH, BATCH_SIZE)
        logger.save_setting(args)
    else:
        logger=None

    SingleDistillation_main(logger)
