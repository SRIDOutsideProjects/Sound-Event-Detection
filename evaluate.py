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
from utilities.evaluation_measures_tf import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, \
    audio_tagging_results, get_f_measure_by_class, get_f1_sed_score, bootstrap, get_psds_ct, get_f1_psds

if __name__ == '__main__':
    batch_nbs = [5,5]
    pooling_time_ratio = 4
    #add_axis_conv = 0
    out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio

    crnn = CRNN()
    crnn_ema = CRNN()
    crnn.build_CRNN_model()
    crnn_ema.build_CRNN_model()

    many_hot_encoder = ManyHotEncoder(labels=cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio, )
    encod_func = many_hot_encoder.encode_strong_df
    weights_file='weights/crnn_final_model.h5'
    valid_df=pd.read_csv('E:/intern/project2/learn/dataset/audio/validation/validation/validation.tsv',sep='\t')
    durations_valid_synth=pd.read_csv('E:/intern/project2/learn/dataset/metadata/validation/validation_durations.tsv',sep='\t')
    mean_ = np.load('weights/mean.npy')
    std_ = np.load('weights/std.npy')
    scaler = Scaler()
    scaler.mean_ = mean_
    scaler.std_ = std_
    transforms = get_transforms(cfg.max_frames, scaler)
    crnn.model.load_weights(weights_file)
    valid_strong = dataloader(valid_df,encod_func,transforms,return_indexes = True)
    valid_strong.create_tf_data_obj()
    dataset1 = valid_strong.tf_data_set.batch(batch_nbs[0], drop_remainder=True)
    validation_labels_df = valid_df.drop("feature_filename", axis=1)
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    predictions = get_predictions(crnn.model, dataset1, valid_df, many_hot_encoder.decode_strong, pooling_time_ratio,
                                          median_window=median_window, save_predictions=None)
    valid_synth = valid_df.drop("feature_filename", axis=1)
    valid_synth_f1, lvf1, hvf1 = bootstrap(predictions, valid_synth, get_f1_sed_score)

    psds_f1_valid, lvps, hvps = bootstrap(predictions, valid_synth, get_f1_psds, meta_df=durations_valid_synth)
    print(f"F1 event_based: {valid_synth_f1}, +- {max(valid_synth_f1-lvf1, hvf1 - valid_synth_f1)},\n"
                f"Psds ct: {psds_f1_valid}, +- {max(psds_f1_valid - lvps, hvps - psds_f1_valid)}")

    predictions.to_csv('outputs/predictions.csv')