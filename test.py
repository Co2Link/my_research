import torch
import os
import numpy as np
import csv
import random
from collections import namedtuple
from agents.base import MemoryStorer

from agents.evaluator import Evaluator
from agents.ddqn import DDQN
from atari_wrappers import *


curriculum_params={
    "prediction_steps": [1, 3, 5],
    "lr": [1e-4, 1e-5, 1e-5],
    "n_epoches": [int(1e3), int(2 * 1e3), int(2 * 1e3)],
}

print(str(curriculum_params))