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

n_channel = 1
add_axis_conv = 0
train_cnn = True
n_layers = 7
classes = cfg.classes

class dataloader():
    def __init__(self, df, encode_function=None,transform=None,return_indexes=False,training = False):
        self.df = df
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes
        self.training = training
    
    def create_generator(self):
        if self.training:
            df = shuffle(self.df)
        else:
            df = self.df
        feat_filenames = df.feature_filename.drop_duplicates()
        filenames = df.filename.drop_duplicates()
        for index,filename in enumerate(feat_filenames):
            features = np.load(filename)
            if 'event_labels' in df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
                if "event_labels" in self.df.columns:
                    label = df.iloc[index]["event_labels"]
                    if pd.isna(label):
                        label = []
                    if type(label) is str:
                        if label == "":
                            label = []
                        else:
                            label = label.split(",")
                else:
                    cols = ["onset", "offset", "event_label"]
                    label = df[df.filename == filenames.iloc[index]][cols]
                    if label.empty:
                        label = []
            else:
                label = "empty"
                if "filename" not in self.df.columns:
                    raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(df.columns))
            
            if self.encode_function is not None:
                y = self.encode_function(label)
            else:
                y = label
            sample = features,y
            if self.transform:
                sample = self.transform(sample)
            features,y = sample
            if self.return_indexes==False:
                yield features,y
            else:
                yield (features,y),feat_filenames.index[index]
    
    def create_tf_data_obj(self):
        if self.return_indexes:
            self.tf_data_set = tf.data.Dataset.from_generator(
                                self.create_generator,
                                output_types=((tf.float32, tf.float32),tf.int32),
                                args=None
                                )
        else:
            self.tf_data_set = tf.data.Dataset.from_generator(
                                self.create_generator,
                                output_types=(tf.float32, tf.float32),
                                args=None
                                )