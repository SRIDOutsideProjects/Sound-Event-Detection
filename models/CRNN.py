import tensorflow as tf
import numpy as np
import pandas as pd
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms_tf import get_transforms
import config as cfg

from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D, Concatenate
import tensorflow.keras.activations as activations
from tensorflow.keras.losses import BinaryCrossentropy,MeanSquaredError
import tensorflow as tf
import keras
import config as cfg
import os
from random import seed
from sklearn.utils import shuffle
import numpy as np
from utilities.Scaler_tf import ScalerPerAudio, Scaler
import soundfile
import librosa
import pandas as pd
from dcase_util.data import ProbabilityEncoder
from dcase_util.data import DecisionEncoder
import time
from tensorflow.contrib import lite

n_channel = 1
add_axis_conv = 0
train_cnn = True
n_layers = 7
classes = cfg.classes

class CRNN():
    def __init__(self, n_in_channel = 1, time_len = 628, freq=128, nclass = len(cfg.classes), attention=False, n_RNN_cell=64, n_layers_RNN=1,
                 activation="relu", dropout=0, kernel_size = n_layers * [3], padding = n_layers * ['same'],
                 stride = n_layers * [1], nb_filters = [16,  32,  64,  128,  128, 128, 128],
                 pooling = [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
                 train_cnn=True, rnn_type='BGRU', dropout_recurrent=0,
                 **kwargs):
        
        self.model = []
        self.time_len = time_len
        self.n_in_channel = n_in_channel
        self.freq = freq
        self.nclass = nclass
        self.attention = attention
        self.n_RNN_cell = n_RNN_cell
        self.n_layers_RNN = n_layers_RNN
        self.activation = activation
        self.dropout = dropout
        self.dropout_recurrent = dropout_recurrent
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.nb_filters = nb_filters
        self.pooling = pooling
                
    def tfBidirectionalGRU(self,x,n_hidden,dropout=0,num_layers=2,return_sequence=True):
        rnn=x
        for i in range(num_layers):
            rnn = layers.GRU(n_hidden,dropout=dropout,return_sequences=True)(rnn)
        return rnn
    
    def tfBidirectionalGRU_stateful(self,x,n_hidden,dropout=0,num_layers=2,return_sequence=True):
        rnn=x
        for i in range(num_layers):
            rnn = layers.GRU(n_hidden,dropout=dropout,unroll=True,return_sequences=True)(rnn)
        return rnn
    
    def tfGLU(self,x):
        inp_shape = x.shape[-1]
        lin = layers.Dense(inp_shape)(x)
        sig = Activation('sigmoid')(x)
        res = layers.Multiply()([lin,sig])
        return res
        
    def tfContextGating(self,x):
        inp_shape=x.shape[-1]
        lin = layers.Dense(inp_shape)(x)
        sig = Activation('sigmoid')(lin)
        res = layers.Multiply()([lin,sig])
        return res
    
    def conv(self, i, batchNormalization, dropout, activ):
        cnn=[]
        nOut = self.nb_filters[i]
        cnn.append(layers.Conv2D(nOut, self.kernel_size[i], self.stride[i], self.padding[i]))
        if batchNormalization:
            cnn.append(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        if activ.lower() == "leakyrelu":
            cnn.append(Activation('leakyrelu'))
        elif activ.lower() == "relu":
            cnn.append(Activation('relu'))
        elif activ.lower() == "glu":
            cnn.append(self.tfGLU)
        elif activ.lower() == "cg":
            cnn.append(self.tfContextGating)
        if dropout is not None:
            cnn.append(layers.Dropout(dropout))
        return cnn
    
    def CNN(self,x, conv_dropout=0):
        total=[]
        batch_norm = True
        for i in range(len(self.nb_filters)):
            total.extend(self.conv(i, batch_norm, conv_dropout, self.activation))
            total.append(layers.AveragePooling2D(self.pooling[i]))
        out=x
        for layer in total:
            out=layer(out)
        return out
    
    def build_CRNN_model(self):
        
        # input layer for time signal
        x = Input(batch_shape=(None, self.time_len, self.freq, self.n_in_channel))
        
        x_cnn = self.CNN(x)
        print(x_cnn.shape)
        bs, frames, freq, chan = x_cnn.shape
        
        x_shape = layers.Reshape((frames, freq*chan))(x_cnn)
            
        x_den1 = layers.Dense(self.nb_filters[-1])(x_shape)
        
        x_rnn = self.tfBidirectionalGRU(x_den1,self.n_RNN_cell, dropout=self.dropout_recurrent, num_layers=self.n_layers_RNN)
        
        x_drop = layers.Dropout(self.dropout)(x_rnn)
        
        print(x_drop.shape)
        
        strong = layers.Dense(self.nclass)(x_drop)  # [bs, frames, nclass]
        strong_act = Activation('sigmoid')(strong)
        
        print(strong_act.shape)
        
        if self.attention:
            sof = layers.Dense(self.nclass)(strong_act)  
            sof_sof = Activation('softmax')(sof)
            sof_clip = tf.clip_by_value(sof_sof, clip_value_min=1e-7, clip_value_max=1)
            weak = tf.math.reduce_sum(strong_act * sof_clip,axis=-2)/tf.math.reduce_sum(sof_clip,axis=-2)
            print(weak.shape)
        else:
            weak = layers.AveragePooling1D(int(strong_act.shape[1]))(strong_act)
            print(weak.shape)
        weak_final = layers.Reshape((self.nclass,))(weak)
        
        self.model = Model(inputs=x, outputs=[strong_act,weak_final])
    
    def build_CRNN_model_stateful(self):
        
        # input layer for time signal
        x = Input(batch_shape=(1, self.time_len, self.freq, self.n_in_channel))
        
        x_cnn = self.CNN(x)
        print(x_cnn.shape)
        bs, frames, freq, chan = x_cnn.shape
        
        x_shape = layers.Reshape((frames, freq*chan))(x_cnn)
            
        x_den1 = layers.Dense(self.nb_filters[-1])(x_shape)
        
        x_rnn = self.tfBidirectionalGRU_stateful(x_den1,self.n_RNN_cell, dropout=self.dropout_recurrent, num_layers=self.n_layers_RNN)
        
        x_drop = layers.Dropout(self.dropout)(x_rnn)
        
        print(x_drop.shape)
        
        strong = layers.Dense(self.nclass)(x_drop)  # [bs, frames, nclass]
        strong_act = Activation('sigmoid')(strong)
        
        print(strong_act.shape)
        
        if self.attention:
            sof = layers.Dense(self.nclass)(strong_act)  
            sof_sof = Activation('softmax')(sof)
            sof_clip = tf.clip_by_value(sof_sof, clip_value_min=1e-7, clip_value_max=1)
            weak = tf.math.reduce_sum(strong_act * sof_clip,axis=-2)/tf.math.reduce_sum(sof_clip,axis=-2)
            print(weak.shape)
        else:
            weak = layers.AveragePooling1D(int(strong_act.shape[1]))(strong_act)
            print(weak.shape)
        weak_final = layers.Reshape((self.nclass,))(weak)
        
        self.model = Model(inputs=x, outputs=[strong_act,weak_final])
        
    
    def create_tf_lite_model(self, weights_file, target_name, use_dynamic_range_quant=False):

        self.build_CRNN_model_stateful()
        self.model.load_weights(weights_file)
        self.model.save('weights/stateful_crnn_final_model.h5')
        converter = lite.TFLiteConverter.from_keras_model_file('weights/stateful_crnn_final_model.h5')
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        # Save the model.
        with open(target_name + '.tflite', 'wb') as f:
          f.write(tflite_model)
