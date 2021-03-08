from __future__ import division
from __future__ import print_function

from   tensorflow.python.keras import layers

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(layers.Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def call(self, inputs):
        ids, num_samples = inputs
        ids = tf.cast(ids,tf.int32)
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)

        # same order
        adj_t = tf.transpose(adj_lists)

        indices = tf.range(start=0,limit=tf.shape(adj_t)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_adj_t = tf.gather(adj_t, shuffled_indices)
        adj_lists = tf.transpose(shuffled_adj_t)

        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])

        return adj_lists
