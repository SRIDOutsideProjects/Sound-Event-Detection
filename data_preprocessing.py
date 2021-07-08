import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

from data_utils.Desed import DESED
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms

if __name__ == '__main__':
    durations_validation = get_durations_df(cfg.validation)

    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                        compute_log=False, use_multiprocessing=True)

    # Always take weak since we need the validation part
    weak_df = dataset.initialize_and_get_df(cfg.weak)
    print('weak labeled dataset created')
    strong_df = dataset.initialize_and_get_df(cfg.validation)
    print('strong labeled dataset created')
    unlabel_df = dataset.initialize_and_get_df(cfg.unlabel)
    print('unlabeled dataset created')