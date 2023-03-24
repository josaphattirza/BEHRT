import sys 

sys.path.append('/home/josaphat/Desktop/research/BEHRT')

from torch.utils.data import DataLoader
import pandas as pd
from common.common import create_folder,H5Recorder
import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert

from model import optimiser
import sklearn.metrics as skm
import math
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch
import time
from sklearn.metrics import roc_auc_score
from common.common import load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction
import warnings
warnings.filterwarnings(action='ignore')

file_config = {
    'vocab':'token2idx-added',  # vocab token2idx idx2token
    'train': './behrt_format_mimic4ed_month_based_train/',
    'test': './behrt_format_mimic4ed_month_based_test/',
}

optim_config = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

global_params = {
    'batch_size': 256,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
    'output_dir': '', # output folder
    'best_name': '',  # output model name
    'max_len_seq': 100,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5
}

pretrainModel = 'exp-model/minvisit3-monthbased-model'   # pretrained MLM path

