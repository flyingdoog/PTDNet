from    inits import *
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.python.keras import layers
from    config import args, params, dtype




# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res




class Dense(layers.Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, bias=False, activation=tf.nn.relu,featureless = False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.type = 'dense'
        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.dropout = dropout

        # helper variable for sparse dropout
        self.weights_ = []
        w = self.add_weight('weight', [input_dim, output_dim],dtype=dtype)
        self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight('bias', [output_dim],dtype=dtype)


    def call(self, inputs, training= None):
        x = inputs

        if tf.__version__.startswith('2.'):
            x = tf.nn.dropout(x, self.dropout)
        else:
            x = tf.nn.dropout(x, 1.0 - self.dropout)

        # transform
        output = dot(x, self.weights_[0])

        # bias
        if self.bias:
            output += self.bias

        return self.act(output)


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.type = 'gcn'

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.usebias = bias
        self.output_dim = output_dim
        self.weights_ = []
        for i in range(1):
            w = self.add_weight('weight' + str(i), [input_dim, output_dim],dtype=dtype)
            self.weights_.append(w)
        if self.usebias:
            self.bias = self.add_weight('bias', [output_dim],dtype=dtype)


    def call(self, inputs, training=None):

        x, support_ = inputs

        # dropout
        if training is not False:
            if tf.__version__.startswith('2.'):
                x = tf.nn.dropout(x, self.dropout)
            else:
                x = tf.nn.dropout(x, 1.0 - self.dropout)

        # convolve
        pre_sup = dot(x, self.weights_[0], sparse=self.is_sparse_inputs)
        output = dot(support_, pre_sup, sparse=False)

        # bias
        if self.usebias:
            output += self.bias

        return self.activation(output)
