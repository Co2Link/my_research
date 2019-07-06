import os
import shutil
from collections import deque
import time
import tensorflow as tf
import numpy as np
import argparse
import csv

root='./root'

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', '--max_iteration', type=int, default=1000000)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('-e_start', '--eps_start', type=float, default=1.0)
    parser.add_argument('-e_end', '--eps_end', type=float, default=0.01)
    parser.add_argument('-e_step', '--eps_step', type=float, default=1e5)
    parser.add_argument('-m_len', '--max_mem_len', type=int, default=10000)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('-s', '--save_model_interval', type=int, default=100)
    parser.add_argument('-r', '--root_path', type=str, default="./root")
    parser.add_argument('-a_type', '--augment_type', type=str, default="None")
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--log', action="store_true")
    args = parser.parse_args()

    with open(os.path.join(root,'csv','setting.csv'),'a',newline='') as f:
        writer=csv.writer(f)
        for k,v in vars(args).items():
            writer.writerow((k,v))

