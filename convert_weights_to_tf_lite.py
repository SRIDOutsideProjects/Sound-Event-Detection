import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

from data_utils.Desed import DESED
from data_utils.DataLoad import dataloader
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler_tf import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms_tf import get_transforms

quantization = False
weights_file='weights/crnn_final_model.h5'
target_folder='weights/model_final_lite'
converter = CRNN()
converter.create_tf_lite_model(weights_file, 
                                  target_folder,
                                  use_dynamic_range_quant=bool(quantization))