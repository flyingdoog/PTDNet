from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inits import zeros

flags = tf.app.flags
FLAGS = flags.FLAGS
from    tensorflow.python.keras import layers

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))



class Dense(layers.Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, num_features_nonzero = None, bias=False,featureless = False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.type = 'dense'
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.dropout = dropout


        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero
        self.weights_ = []
        w = self.add_weight('weight', [input_dim, output_dim],dtype=tf.float32)
        self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight('bias', [output_dim],dtype=tf.float32)


    def call(self, inputs, training= None):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        else:
            if tf.__version__.startswith('2.'):
                x = tf.nn.dropout(x, self.dropout)
            else:
                x = tf.nn.dropout(x, 1.0 - self.dropout)

        # transform
        output = dot(x, self.weights_[0], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.bias

        return self.act(output)