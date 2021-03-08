import tensorflow as tf
import numpy as np
from    tensorflow.python.keras import layers
from config import *


class Attention(layers.Layer):
    def __init__(self, rows_tile=None, **kwargs):
        super(Attention, self).__init__(**kwargs)

        # self.rows_tile = tf.convert_to_tensor(rows_tile,dtype=tf.int64)
        hidden_1 = args.hidden_1
        hidden_2 = args.hidden_2

        if args.act=='leaky_relu':
            act = tf.nn.leaky_relu
        elif args.act == 'relu':
            act = tf.nn.relu
        elif args.act =='sigmoid':
            act = tf.nn.sigmoid
        elif args.act =='tanh':
            act = tf.nn.tanh
        else:
            act = lambda x:x

        if args.initializer=='he':
            initializer = 'he_normal'
        else:
            initializer = tf.initializers.glorot_normal()#

        self.nblayer = layers.Dense(hidden_1, activation=act, kernel_initializer = initializer,dtype=dtype)
        self.selflayer = layers.Dense(hidden_1, activation=act , kernel_initializer =initializer,dtype=dtype)

        self.attention = []
        if hidden_2>0:
            self.attention.append(layers.Dense(hidden_2, activation=act , kernel_initializer = initializer,dtype=dtype))

        self.attention.append(layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer,dtype=dtype))

        self.attention_layers = []
        self.attention_layers.append(self.nblayer)
        self.attention_layers.append(self.selflayer)

        for layer in self.attention:
            self.attention_layers.append(layer)

    def call(self,input1,input2,training=False):
        nb_layer = self.nblayer
        selflayer = self.selflayer
        nn = self.attention

        batch_size = input1.shape[0]
        nbs_num = input1.shape[1]

        if tf.__version__.startswith('2.'):
            dp = args.dropout2
        else:
            dp = 1-args.dropout2

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1,dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2,dp)

        x_feature_expand = tf.expand_dims(input2, axis=1)
        x_feature_tile = tf.tile(x_feature_expand,[1,nbs_num,1])
        x_feature_tile = tf.reshape(x_feature_tile,(batch_size*nbs_num, args.hidden_1))
        nbs_feature_tile = tf.reshape(input1,(batch_size*nbs_num, args.hidden_1))

        input10 = tf.concat([nbs_feature_tile, x_feature_tile], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]

        return weight10