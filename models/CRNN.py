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
import torch.nn as nn
import torch
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

n_channel = 1
add_axis_conv = 0
train_cnn = True
n_layers = 7
classes = cfg.classes

class CRNN():
    def __init__(self, n_in_channel = 1, time_len = 628, freq=128, nclass = len(cfg.classes), attention=False, n_RNN_cell=64, n_layers_RNN=1,
                 activation="leakyrelu", dropout=0, kernel_size = n_layers * [3], padding = n_layers * ['same'],
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
        os.environ['PYTHONHASHSEED']=str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.x
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)
                
    def tfBidirectionalGRU(self,x,n_hidden,dropout=0,num_layers=2,return_sequence=True):
        rnn=x
        for i in range(num_layers):
            rnn = layers.Bidirectional(tf.keras.layers.GRU(n_hidden,dropout=dropout,return_sequences=True))(rnn)
        return rnn
    
    def tfBidirectionalGRU_stateful(self,x,in_states,n_hidden,dropout=0,num_layers=2,return_sequence=True):
        rnn=x
        states_h1=[]
        states_h2=[]
        for idx in range(num_layers):
            in_state = [in_states[:,idx,:, 0], in_states[:,idx,:, 1]]
            rnn,h_state1, h_state2 = layers.Bidirectional(tf.keras.layers.GRU(
                n_hidden,dropout=dropout, unroll = True, return_sequences=True, return_state=True))(rnn, initial_state=in_state)
            states_h1.append(h_state1)
            states_h2.append(h_state2)
        out_states_h1 = tf.reshape(tf.stack(states_h1, axis=0),
                                  [1,num_layers,n_hidden])
        out_states_h2 = tf.reshape(tf.stack(states_h2, axis=0),
                                   [1,num_layers,n_hidden])
        out_states = tf.stack([out_states_h1, out_states_h2], axis=-1)
        return rnn,out_states
    
    def tfGLU(self,x):
        inp_shape = x.shape[-1]
        lin = layers.Dense(inp_shape)(x)
        sig = activations.sigmoid(x)
        res = lin * sig
        return res
        
    def tfContextGating(self,x):
        inp_shape=x.shape[-1]
        lin = layers.Dense(inp_shape)(x)
        sig = activations.sigmoid(lin)
        res = lin * sig
        return res
    
    def conv(self, i, batchNormalization, dropout, activ):
        cnn=[]
        nOut = self.nb_filters[i]
        cnn.append(layers.Conv2D(nOut, self.kernel_size[i], self.stride[i], self.padding[i],kernel_initializer = tf.keras.initializers.GlorotNormal()))
        if batchNormalization:
            cnn.append(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
        if activ.lower() == "leakyrelu":
            cnn.append(layers.LeakyReLU())
        elif activ.lower() == "relu":
            cnn.append(activations.relu)
        elif activ.lower() == "glu":
            cnn.append(tfGLU)
        elif activ.lower() == "cg":
            cnn.append(tfContextGating)
        if dropout is not None:
            cnn.append(layers.Dropout(dropout))
        return cnn
    
    def CNN(self,x, activation="relu", conv_dropout=0):
        total=[]
        batch_norm = True
        for i in range(len(self.nb_filters)):
            total.extend(self.conv(i, batch_norm, conv_dropout, activation))
            total.append(layers.AveragePooling2D(self.pooling[i]))
        out=x
        for layer in total:
            out=layer(out)
        return out
    
    def build_CRNN_model(self):
        
        # input layer for time signal
        x = Input(batch_shape=(None, None, self.freq, self.n_in_channel))
        
        x_cnn = self.CNN(x)
        print(x_cnn.shape)
        bs, frames, freq, chan = x_cnn.shape
        if freq!=1:
            x_shape=tf.reshape(x_cnn,(bs, frames, freq*chan))
        else:
            x_shape=tf.squeeze(x_cnn,(2))
            
        x_den1 = layers.Dense(self.nb_filters[-1],kernel_initializer = tf.keras.initializers.GlorotNormal())(x_shape)
        
        x_rnn = self.tfBidirectionalGRU(x_den1,self.n_RNN_cell, dropout=self.dropout_recurrent, num_layers=self.n_layers_RNN)
        
        x_drop = layers.Dropout(self.dropout)(x_rnn)
        
        print(x_drop.shape)
        
        strong = layers.Dense(self.nclass,kernel_initializer = tf.keras.initializers.GlorotNormal())(x_drop)  # [bs, frames, nclass]
        strong_act = activations.sigmoid(strong)
        
        print(strong_act.shape)
        
        if self.attention:
            sof = layers.Dense(self.nclass,kernel_initializer = tf.keras.initializers.GlorotNormal())(strong_act)  
            sof_sof = activations.softmax(sof)
            sof_clip = tf.clip_by_value(sof_sof, clip_value_min=1e-7, clip_value_max=1)
            weak = tf.math.reduce_sum(strong_act * sof_clip,axis=-2)/tf.math.reduce_sum(sof_clip,axis=-2)
            print(weak.shape)
        else:
            weak = tf.math.reduce_mean(strong_act, axis=-2)
            print(weak.shape)
        
        self.model = Model(inputs=x, outputs=[strong_act,weak])
    
    def build_CRNN_model_stateful(self):
        
        # input layer for time signal
        x = Input(batch_shape=(1, self.time_len, self.freq, self.n_in_channel))
        
        x_cnn = self.CNN(x)
        print(x_cnn.shape)
        bs, frames, freq, chan = x_cnn.shape
        if freq!=1:
            x_shape=tf.reshape(x_cnn,(bs, frames, freq*chan))
        else:
            x_shape=tf.squeeze(x_cnn,(2))
            
        x_den1 = layers.Dense(self.nb_filters[-1],kernel_initializer = tf.keras.initializers.GlorotNormal())(x_shape)
        
        x_rnn = self.tfBidirectionalGRU(x_den1,self.n_RNN_cell, dropout=self.dropout_recurrent, num_layers=self.n_layers_RNN)
        
        x_drop = layers.Dropout(self.dropout)(x_rnn)
        
        print(x_drop.shape)
        
        strong = layers.Dense(self.nclass,kernel_initializer = tf.keras.initializers.GlorotNormal())(x_drop)  # [bs, frames, nclass]
        strong_act = activations.sigmoid(strong)
        
        print(strong_act.shape)
        
        if self.attention:
            sof = layers.Dense(self.nclass,kernel_initializer = tf.keras.initializers.GlorotNormal())(strong_act)  
            sof_sof = activations.softmax(sof)
            sof_clip = tf.clip_by_value(sof_sof, clip_value_min=1e-7, clip_value_max=1)
            weak = tf.math.reduce_sum(strong_act * sof_clip,axis=-2)/tf.math.reduce_sum(sof_clip,axis=-2)
            print(weak.shape)
        else:
            weak = tf.math.reduce_mean(strong_act, axis=-2)
            print(weak.shape)
        
        self.model = Model(inputs=x, outputs=[strong_act,weak])
    
    def compile_model(self):
        
        self.optimizerAdam = keras.optimizers.Adam(lr=cfg.default_learning_rate, clipnorm=3.0)
        self.model.compile()
        
    
    def create_tf_lite_model(self, weights_file, target_name, use_dynamic_range_quant=False):
        
        self.build_CRNN_model_stateful()
        self.model.load_weights(weights_file)
        
        x = Input(batch_shape=(1, self.time_len, self.freq, self.n_in_channel))
        states_in_1 = Input(batch_shape=(1,self.n_layers_RNN,self.n_RNN_cell,2))
        x_cnn = self.CNN(x)
        bs, frames, freq, chan = x_cnn.shape
        x_shape=tf.reshape(x_cnn,(bs,frames, freq*chan))
        x_den1 = layers.Dense(self.nb_filters[-1])(x_shape)
        print('xden',x_den1.shape)
        x_rnn,states_out_1 = self.tfBidirectionalGRU_stateful(x_den1,states_in_1,self.n_RNN_cell, dropout=self.dropout_recurrent, num_layers=self.n_layers_RNN)
        print(states_out_1.shape)
        x_drop = layers.Dropout(self.dropout)(x_rnn)
        strong = layers.Dense(self.nclass)(x_drop)  # [bs, frames, nclass]
        strong_act = activations.sigmoid(strong)
        print('strong:',strong.shape)
        if self.attention:
            sof = layers.Dense(self.nclass)(strong_act)  
            sof_sof = activations.softmax(sof)
            sof_clip = tf.clip_by_value(sof_sof, clip_value_min=1e-7, clip_value_max=1)
            weak = tf.math.reduce_sum(strong_act * sof_clip,axis=-2)/tf.math.reduce_sum(sof_clip,axis=-2)
        else:
            weak = tf.math.reduce_mean(strong_act, axis=-2)
        
        print('weak:',weak.shape)
        model_lite = Model(inputs=[x,states_in_1], outputs=[strong_act,weak,states_out_1])
        
        weights = self.model.get_weights()
        model_lite.set_weights(weights)
        converter = tf.lite.TFLiteConverter.from_keras_model(model_lite)
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '.tflite', 'wb') as f:
              f.write(tflite_model)