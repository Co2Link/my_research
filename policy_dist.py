import argparse
from logWriter import LogWriter
from keras import backend as K
import tensorflow as tf
import time
import glob
import csv
import os
from atari_wrappers import make_atari,wrap_deepmind

from agents.teacher import Teacher
from agents.student import SingleDtStudent
from agents.evaluator import Evaluator


def SingleDistillation_main():

    logger = LogWriter(ROOT_PATH, BATCH_SIZE)
    logger.save_setting(args)

    env = make_atari(GAME)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    teacher = Teacher(MODEL_PATH, env, EPSILON, MEM_SIZE)

    student = SingleDtStudent(env, LEARNING_RATE, logger, BATCH_SIZE, EPSILON, teacher, ADD_MEM_NUM, UPDATE_NUM, EPOCH,
                              LOSS_FUC)

    student.distill()

    logger.save_model(student, 'student_{}'.format(LOSS_FUC))
    logger.log_total_time_cost()


def Evaluation():
    """
    evaluation the performance of both teacher and students under Single-target-distillation situation
    there should be only one model file and csv file under the directory of './model/teacher'
    multiple log directory under path of './result_DT'
    """

    # get the game_name from setting.csv
    with open(glob.glob('./model/teacher/*.csv')[0]) as f:
        reader = csv.reader(f)
        settings_dict = {row[0]: row[1] for row in reader}
    game_name = settings_dict['game']
    print("*** GAME of teacher:{} ***".format(game_name))

    # make environment
    env = make_atari(GAME)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    # log
    root='result_EVAL'
    if not os.path.exists(root):
        os.mkdir(root)
        print('*** Create folder: {} ***'.format(root))
    now_time = time.strftime('%y%m%d_%H%M%S', time.localtime())
    save_path = os.path.join(root, now_time).replace('\\', '/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('*** Create folder: {} ***'.format(save_path))

    # evaluate teacher
    Teacher(glob.glob('./model/teacher/*.h5f')[0].replace('\\', '/'), env, eval_iteration=EVAL_ITERATION).evaluate(save_path)

    # evaluate student
    for log_path in glob.glob('./result_DT/*'):
        Evaluator(env, log_path.replace('\\', '/'),save_path,eval_iteration=EVAL_ITERATION).evaluate()


def test():
    """ test distillation and evaluation """
    LEARNING_RATE = 0.0001
    GAME = 'BreakoutNoFrameskip-v4'
    BATCH_SIZE = 32
    EPSILON = 0.05
    ADD_MEM_NUM = 3000
    UPDATE_NUM = 200
    EPOCH = 1
    MEM_SIZE = 50000
    MODEL_PATH = './model/teacher/breakout-1.h5f'
    LOSS_FUC = 'mse'
    EVAL_ITERATION = 3000

    logger = LogWriter(ROOT_PATH, BATCH_SIZE)
    logger.save_setting(args)

    env = make_atari(GAME)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    teacher = Teacher(MODEL_PATH, env, EPSILON, MEM_SIZE, EVAL_ITERATION)

    student = SingleDtStudent(env, LEARNING_RATE, logger, BATCH_SIZE, EPSILON, teacher, ADD_MEM_NUM, UPDATE_NUM, EPOCH,
                              LOSS_FUC)

    student.distill()

    logger.save_model(student, 'student_{}'.format(LOSS_FUC))
    logger.log_total_time_cost()

    # log
    root='result_EVAL'
    if not os.path.exists(root):
        os.mkdir(root)
        print('*** Create folder: {} ***'.format(root))
    now_time = time.strftime('%y%m%d_%H%M%S', time.localtime())
    save_path = os.path.join(root, now_time).replace('\\', '/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('*** Create folder: {} ***'.format(save_path))


    # evaluate teacher
    teacher.evaluate(save_path)

    # evaluate student
    for log_path in glob.glob('./result_DT/*'):
        Evaluator(env, log_path,save_path, eval_iteration=EVAL_ITERATION).evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-g', '--game', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.05)
    parser.add_argument('--add_mem_num', type=int, default=int(5e3))
    parser.add_argument('--update_num', type=int, default=int(1e3))
    parser.add_argument('-ep', '--epoch', type=int, default=20)
    parser.add_argument('--mem_size', type=int, default=int(5e4))
    parser.add_argument('-r', '--root_path', type=str, default='./result_DT')
    parser.add_argument('--model_path', type=str, default='./model/teacher/breakout-1.h5f')
    parser.add_argument('--loss_fuc', type=str, default='mse')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-eval', '--evaluate', action='store_true')
    parser.add_argument('-dt', '--distillate', action='store_true')
    parser.add_argument('--eval_iteration', type=int, default=int(1e5))
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
    LOSS_FUC = args.loss_fuc
    EVAL_ITERATION = args.eval_iteration

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    K.set_session(sess)



    if args.test:
        test()
    elif args.distillate:
        SingleDistillation_main()
    elif args.evaluate:
        Evaluation()
