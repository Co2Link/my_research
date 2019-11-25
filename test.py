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

from model_based import test_predict,test_predict_multi


model_path = "result_WORLD/191123_200420/models/model_5.pt"
# test_predict(model_path=model_path)
test_predict_multi(model_path=model_path,prediction_step=10)