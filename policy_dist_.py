import argparse
import os
import csv
import torch

from logWriter import LogWriter
from atari_wrappers import make_atari, wrap_deepmind
from agents.teacher_ import Teacher
from agents.student_ import SingleDtStudent

torch.set_num_threads(1)

def SingleDistillation_main():
    logger = LogWriter(ROOT_PATH)
    logger.save_setting(vars(args))

    with open(os.path.join(SOURCE_LOG_PATH, 'setting.csv'), 'r') as f:
        setting_dict = {row[0]: row[1] for row in csv.reader(f)}
        game_name = setting_dict['game']

    env = make_atari(game_name)
    env = wrap_deepmind(env, frame_stack=True)
    teacher_hparams = {'n_actions': env.action_space.n,
                       'net_size': setting_dict['net_size'], 'state_shape': env.observation_space.shape,'eps':EPSILON,'mem_size':MEM_SIZE}
    teacher = Teacher(load_model_path=os.path.join(
        SOURCE_LOG_PATH, 'models', 'model_final.pt'), hparams=teacher_hparams, env=env)
    student_hparams = {'n_actions': env.action_space.n, 'net_size': TARGET_NET_SIZE, 'state_shape': env.observation_space.shape, 'lr': LEARNING_RATE, 'epoch': EPOCH, 'add_mem_num': ADD_MEM_NUM, 'n_update':N_UPDATE}
    student = SingleDtStudent(logger, student_hparams)
    student.distill(teacher)

    logger.save_model(student)


def test():
    pass

def Evaluation():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.05)
    parser.add_argument('--add_mem_num', type=int, default=int(5e3))
    parser.add_argument('--n_update', type=int, default=int(1e3))
    parser.add_argument('-ep', '--epoch', type=int, default=100)
    parser.add_argument('--mem_size', type=int, default=int(5e4))
    parser.add_argument('-r', '--root_path', type=str, default='./result_DT')
    parser.add_argument('--source_log_path', type=str,
                        default='result/191119_214214')
    parser.add_argument('--target_net_size', type=str, default='normal')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-eval', '--evaluate', action='store_true')
    parser.add_argument('-dt', '--distillate', action='store_true')
    parser.add_argument('--eval_iteration', type=int, default=int(1e5))
    args = parser.parse_args()

    LEARNING_RATE = args.learning_rate
    EPSILON = args.epsilon
    ADD_MEM_NUM = args.add_mem_num
    N_UPDATE = args.n_update
    EPOCH = args.epoch
    MEM_SIZE = args.mem_size
    ROOT_PATH = args.root_path
    SOURCE_LOG_PATH = args.source_log_path
    EVAL_ITERATION = args.eval_iteration
    TARGET_NET_SIZE = args.target_net_size

    if args.test:
        test()
    elif args.distillate:
        SingleDistillation_main()
    elif args.evaluate:
        Evaluation()
