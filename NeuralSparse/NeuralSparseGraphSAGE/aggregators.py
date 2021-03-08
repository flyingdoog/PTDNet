import tensorflow as tf

from    tensorflow.python.keras import layers
from inits import glorot, zeros
from config import *


def sample_gumbel(shape):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, istrain):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    r = sample_gumbel(tf.shape(logits))
    values = tf.cond(istrain is not None, lambda: tf.math.log(logits) + r, lambda: tf.math.log(logits))
    values /= temperature
    y = tf.nn.softmax(values,-1)
    return y


class MeanAggregator(layers.Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, activation=tf.nn.relu, attention=None,
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.use_bias = bias
        self.act = activation
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.neigh_weights = self.add_weight('neigh_weights', [neigh_input_dim, output_dim],dtype=dtype)
        self.self_weights = self.add_weight('self_weights', [input_dim, output_dim],dtype=dtype)
        if self.use_bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=dtype,initializer=tf.zeros_initializer)

        self.weights_ = []
        self.weights_.append(self.neigh_weights)
        self.weights_.append(self.self_weights)

        self.attention = attention

        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs, temperature = inputs
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        edge_weight = self.attention(neigh_vecs,self_vecs,training=training)
        edge_weight = tf.reshape(edge_weight,(neigh_vecs.shape[0],neigh_vecs.shape[1]))
        pi = tf.nn.softmax(edge_weight,-1)
        mask_values = gumbel_softmax_sample(pi, temperature, training)

        # select top k
        top_k_v, top_k_i = tf.math.top_k(mask_values,args.topK)

        kth = tf.reduce_min(top_k_v,-1) # N,
        kth = tf.expand_dims(kth,-1)
        kth = tf.tile(kth,[1,pi.shape[-1]]) # N,K
        mask2 = tf.greater_equal(mask_values, kth)
        mask2 = tf.cast(mask2, tf.float32)
        row_sum = tf.reduce_sum(mask2,-1)

        dense_support = mask2

        if args.weighted:
            dense_support = tf.multiply(mask_values,mask2)
        else:
            print('no gradient bug here!')
            exit()

        dense_support = tf.expand_dims(dense_support,-1)
        masked_neigh_vecs = tf.multiply(dense_support,neigh_vecs)
        neigh_means = tf.reduce_mean(masked_neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.neigh_weights)
        from_neighs = tf.nn.dropout(from_neighs, 1-self.dropout)

        from_self = tf.matmul(self_vecs, self.self_weights)
        from_self = tf.nn.dropout(from_self, 1-self.dropout)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.use_bias:
            output += self.bias

        return self.act(output),edge_weight

class GCNAggregator(layers.Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, activation=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.use_bias = bias
        self.act = activation
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.weights = self.add_weight('neigh_weights', [neigh_input_dim, output_dim],dtype=dtype)
        if self.use_bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=dtype,initializer=tf.zeros_initializer)

        self.weights_ = []
        self.weights_.append(self.weights)
        self.weights_.append(self.self_weights)


        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs, temperature = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        edge_weight = None
        #
        # if args.l0:
        #     edge_weight = self.attention(neigh_vecs,self_vecs,training=training)
        #     mask_values10 = hard_concrete_sample(edge_weight, temperature, training)
        #     mask_values10 = tf.reshape(mask_values10,(neigh_vecs.shape[0],neigh_vecs.shape[1],1))
        #     masked_neigh_vecs = tf.multiply(mask_values10,neigh_vecs)
        #     means = tf.reduce_mean(tf.concat([masked_neigh_vecs,
        #                                       tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)


        # [nodes] x [out_dim]
        output = tf.matmul(means, self.weights)
        output = tf.nn.dropout(output, 1-self.dropout)

        # bias
        if self.use_bias:
            output += self.bias
       
        return self.act(output),edge_weight


class MaxPoolingAggregator(layers.Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, activation=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.use_bias = bias
        self.act = activation
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(layers.Dense(hidden_dim, act=tf.nn.relu)) # dropout,output_dim=,#input_dim=neigh_input_dim,

        self.neigh_weights = self.add_weight('neigh_weights', [hidden_dim, output_dim],dtype=dtype)
        self.self_weights = self.add_weight('self_weights', [input_dim, output_dim],dtype=dtype)
        if self.use_bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=dtype,initializer=tf.zeros_initializer)

        self.weights_ = []
        self.weights_.append(self.neigh_weights)
        self.weights_.append(self.self_weights)


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs, temperature = inputs

        edge_weight=None
        # if args.l0:
        #     edge_weight = self.attention(neigh_vecs,self_vecs,training=training)
        #     mask_values10 = hard_concrete_sample(edge_weight, temperature, training)
        #     mask_values10 = tf.reshape(mask_values10,(neigh_vecs.shape[0],neigh_vecs.shape[1],1))
        #     masked_neigh_vecs = tf.multiply(mask_values10,neigh_vecs)
        #     neigh_vecs = tf.reduce_mean(masked_neigh_vecs, axis=1)

        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        from_neighs = tf.nn.dropout(from_neighs, 1-self.dropout)
        from_self = tf.nn.dropout(from_self, 1-self.dropout)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.use_bias:
            output += self.bias
       
        return self.act(output),edge_weight

class MeanPoolingAggregator(layers.Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, activation=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.use_bias = bias
        self.act = activation
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(layers.Dense(hidden_dim,activation=tf.nn.relu)) # dropout here input_dim=neigh_input_dim,output_dim=

        self.neigh_weights = self.add_weight('neigh_weights', [hidden_dim, output_dim],dtype=dtype)
        self.self_weights = self.add_weight('self_weights', [input_dim, output_dim],dtype=dtype)
        if self.use_bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=dtype,initializer=tf.zeros_initializer)

        self.weights_ = []
        self.weights_.append(self.neigh_weights)
        self.weights_.append(self.self_weights)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs, temperature = inputs

        edge_weight = None
        # if args.l0:
        #     edge_weight = self.attention(neigh_vecs,self_vecs,training=training)
        #     mask_values10 = hard_concrete_sample(edge_weight, temperature, training)
        #     mask_values10 = tf.reshape(mask_values10,(neigh_vecs.shape[0],neigh_vecs.shape[1],1))
        #     masked_neigh_vecs = tf.multiply(mask_values10,neigh_vecs)
        #     neigh_vecs = tf.reduce_mean(masked_neigh_vecs, axis=1)

        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        from_neighs = tf.nn.dropout(from_neighs, 1-self.dropout)
        from_self = tf.nn.dropout(from_self, 1-self.dropout)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.use_bias:
            output += self.bias
       
        return self.act(output),edge_weight


class TwoMaxLayerPoolingAggregator(layers.Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, activation=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = activation
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim


        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(layers.Dense(hidden_dim_1,activation=tf.nn.relu,)) #input_dim=neigh_input_dim,dropout=dropout

        self.mlp_layers.append(layers.Dense(hidden_dim_2, activation=tf.nn.relu))#dropout=dropout


        self.neigh_weights = self.add_weight('neigh_weights', [hidden_dim_2, output_dim],dtype=dtype)
        self.self_weights = self.add_weight('self_weights', [input_dim, output_dim],dtype=dtype)
        if self.use_bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=dtype,initializer=tf.zeros_initializer)

        self.weights_ = []
        self.weights_.append(self.neigh_weights)
        self.weights_.append(self.self_weights)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs, temperature = inputs
        edge_weight = None
        # if args.l0:
        #     edge_weight = self.attention(neigh_vecs,self_vecs,training=training)
        #     mask_values10 = hard_concrete_sample(edge_weight, temperature, training)
        #     mask_values10 = tf.reshape(mask_values10,(neigh_vecs.shape[0],neigh_vecs.shape[1],1))
        #     masked_neigh_vecs = tf.multiply(mask_values10,neigh_vecs)
        #     neigh_vecs = tf.reduce_mean(masked_neigh_vecs, axis=1)


        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        from_neighs = tf.nn.dropout(from_neighs, 1-self.dropout)
        from_self = tf.nn.dropout(from_self, 1-self.dropout)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.use_bias:
            output += self.bias
       
        return self.act(output),edge_weight

class SeqAggregator(layers.Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, activation=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.use_bias = bias
        self.act = activation
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        self.neigh_weights = self.add_weight('neigh_weights', [hidden_dim, output_dim],dtype=dtype)
        self.self_weights = self.add_weight('self_weights', [input_dim, output_dim],dtype=dtype)
        if self.use_bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=dtype,initializer=tf.zeros_initializer)

        self.weights_ = []
        self.weights_.append(self.neigh_weights)
        self.weights_.append(self.self_weights)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def call(self, inputs, training=False):
        self_vecs, neigh_vecs, temperature = inputs

        edge_weight = None
        # if args.l0:
        #     edge_weight = self.attention(neigh_vecs,self_vecs,training=training)
        #     mask_values10 = hard_concrete_sample(edge_weight, temperature, training)
        #     mask_values10 = tf.reshape(mask_values10,(neigh_vecs.shape[0],neigh_vecs.shape[1],1))
        #     masked_neigh_vecs = tf.multiply(mask_values10,neigh_vecs)
        #     neigh_vecs = tf.reduce_mean(masked_neigh_vecs, axis=1)


        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.neigh_weights)
        from_self = tf.matmul(self_vecs, self.self_weights)
        from_neighs = tf.nn.dropout(from_neighs, 1-self.dropout)
        from_self = tf.nn.dropout(from_self, 1-self.dropout)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.use_bias:
            output += self.bias
       
        return self.act(output),edge_weight

