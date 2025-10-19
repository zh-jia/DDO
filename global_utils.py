import random
import numpy as np
import torch as th
from transformers import set_seed
import os

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)