import argparse
import datetime
import inspect
import os
import time
from pprint import pprint
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D, Concatenate
import tensorflow.keras.activations as activations
from tensorflow.keras.losses import BinaryCrossentropy,MeanSquaredError

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

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    new_weights=[]
    for ema_params,params in zip(ema_model.get_weights(),model.get_weights()):
        ema_params=ema_params*alpha + (1-alpha)*params
        new_weights.append(ema_params)
    ema_model.set_weights(new_weights) 
    return ema_model

crnn = CRNN()
crnn_ema = CRNN()
crnn.build_CRNN_model()
crnn_ema.build_CRNN_model()

weak_df=pd.read_csv('E:/intern/project2/learn/dataset/audio/train/weak/weak.tsv',sep='\t')
strong_df=pd.read_csv('E:/intern/project2/learn/dataset/audio/validation/validation/validation.tsv',sep='\t')
unlabel_df=pd.read_csv('E:/intern/project2/learn/dataset/audio/train/unlabel_in_domain/unlabel_in_domain.tsv',sep='\t')

batch_nbs = [5,5]
pooling_time_ratio = 4
#add_axis_conv = 0
out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio

train_weak_df = weak_df.sample(frac=0.9)
valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
train_weak_df = train_weak_df.reset_index(drop=True)
strong_df.onset = strong_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
strong_df.offset = strong_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio

many_hot_encoder = ManyHotEncoder(labels=cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio, )
encod_func = many_hot_encoder.encode_strong_df
transforms = get_transforms(cfg.max_frames )

train_strong = dataloader(strong_df,encod_func,transforms)
train_strong.create_tf_data_obj()
dataset1 = train_strong.tf_data_set.batch(batch_nbs[0], drop_remainder=True)
train_weak = dataloader(train_weak_df,encod_func,transforms)
train_weak.create_tf_data_obj()
dataset2 = train_weak.tf_data_set.batch(batch_nbs[0], drop_remainder=True)

dataset_total = dataset1.concatenate(dataset2)
scaler = Scaler()
scaler.calculate_scaler(dataset_total)
transforms = get_transforms(cfg.max_frames, scaler)

np.save('weights/mean',scaler.mean_)
np.save('weights/std',scaler.std_)

train_strong = dataloader(strong_df,encod_func,transforms,training = True)
train_strong.create_tf_data_obj()
dataset1 = train_strong.tf_data_set.batch(batch_nbs[0], drop_remainder=True).repeat()
train_weak = dataloader(train_weak_df,encod_func,transforms,training = True)
train_weak.create_tf_data_obj()
dataset2 = train_weak.tf_data_set.batch(batch_nbs[0], drop_remainder=True).repeat()

epochs = 10

class_criterion = BinaryCrossentropy()
consistency_criterion = MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=0.005)
c_epoch=0
mask_weak=slice(batch_nbs[0], batch_nbs[0]+batch_nbs[1], 1)
mask_strong=slice(batch_nbs[0])
ema_model = crnn_ema.model

i=0
for c_epoch in range(epochs):
    print("Start of epoch %d" % (c_epoch,))
    for (batch_whole_1,batch_whole_2) in (zip(dataset1,dataset2)):
        strongx,strongy = batch_whole_1
        weakx,weaky = batch_whole_2
        batch_input = tf.concat((strongx,weakx),axis=0)
        #batch_input = tf.expand_dims(batch_input,axis=3)
        print(batch_input.shape)
        target = tf.concat((strongy,weaky),axis=0)
        ema_batch_input=tf.identity(batch_input)
        with tf.GradientTape() as tape:
            strong_pred_ema, weak_pred_ema = crnn_ema.model(ema_batch_input)
            strong_pred_ema = tf.stop_gradient(strong_pred_ema)
            weak_pred_ema = tf.stop_gradient(weak_pred_ema)
            strong_pred, weak_pred = crnn.model(batch_input,training = True)

            loss = None
            target_weak = tf.math.reduce_max(target,axis=-2)
            if mask_weak is not None:
                weak_class_loss = class_criterion(target_weak[mask_weak], weak_pred[mask_weak] )
                ema_class_loss = class_criterion(target_weak[mask_weak],weak_pred_ema[mask_weak])
                loss = weak_class_loss
                print(f'batch: {i} weak_class_loss: {weak_class_loss}')
                
            if mask_strong is not None:
                strong_class_loss = class_criterion(target[mask_strong],strong_pred[mask_strong])
                strong_ema_class_loss = class_criterion(target[mask_strong],strong_pred_ema[mask_strong])
                if loss is not None:
                    loss += strong_class_loss
                    print(f'batch: {i} strong_class_loss: {strong_class_loss}')
                else:
                    loss = strong_class_loss
                    print(f'batch: {i} strong_class_loss: {strong_class_loss}')

            if ema_model is not None:
                consistency_weight = cfg.max_consistency_cost
                consistency_loss_strong = consistency_weight * consistency_criterion(strong_pred, strong_pred_ema)
                #consistency_loss_strong=tf.cast(consistency_loss_strong,dtype=tf.float64)
                if loss is not None:
                    loss += consistency_loss_strong
                    print(f'batch: {i} consistency_loss_strong: {consistency_loss_strong}')
                else:
                    loss = consistency_loss_strong
                consistency_loss_weak = consistency_weight * consistency_criterion(weak_pred, weak_pred_ema)
                #consistency_loss_weak=tf.cast(consistency_loss_weak,dtype=tf.float64)
                if loss is not None:
                    loss += consistency_loss_weak
                    print(f'batch: {i} consistency_loss_weak: {consistency_loss_weak}')
                else:
                    loss = consistency_loss_weak
        grads = tape.gradient(loss, crnn.model.trainable_weights)
        #print(grads)
        optimizer.apply_gradients(
            zip(grads, crnn.model.trainable_weights)
        )
        crnn_ema.model = update_ema_variables(crnn.model,crnn_ema.model,0.999, c_epoch * 100 + i)
        print(f'batch: {i} loss: {loss}')
        print('.......................................')
        i+=1
        if i==100:
            i=0
            optimizer = tf.keras.optimizers.Adam(lr=0.005*(c_epoch+1)*0.9)
            crnn.model.save('weights/crnn_final_model.h5')
            crnn_ema.model.save('weights/crnn_ema_final_model.h5')
            break
    print('epoch done')

