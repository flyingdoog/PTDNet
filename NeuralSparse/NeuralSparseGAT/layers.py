import  tensorflow as tf
from    tensorflow.python.keras import layers
from    config import args, dtype

class attn_head(layers.Layer):
    """attn_head layer."""
    def __init__(self, output_dim, in_drop=0.0, coef_drop=0.0,
                 residual=False, bias=False, activation=tf.nn.relu, **kwargs):
        super(attn_head, self).__init__(**kwargs)
        self.type = 'attn_head'
        self.act = activation
        self.bias = bias
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual

        self.bias = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")

        self.seq_fts_layer = tf.layers.Conv1D(output_dim, kernel_size=1, use_bias=False)
        self.f_1_layer = tf.layers.Conv1D(1, kernel_size=1)
        self.f_2_layer = tf.layers.Conv1D(1, kernel_size=1)
        self.ret_conv_layer = tf.layers.Conv1D(output_dim, kernel_size=1)


    def call(self, inputs, training= None):
        seq, bias_mat = inputs
        with tf.name_scope('my_attn'):
            if training and self.in_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    seq = tf.nn.dropout(seq, self.in_drop)
                else:
                    seq = tf.nn.dropout(seq, 1.0 - self.in_drop)

            # add a new axis
            ext_seq = tf.expand_dims(seq,axis=0)
            ext_seq_fts = self.seq_fts_layer(ext_seq)


            # simplest self-attention possible
            f_1 = self.f_1_layer(ext_seq_fts)
            f_2 = self.f_2_layer(ext_seq_fts)

            ext_logits = f_1 + tf.transpose(f_2, [0, 2, 1])

            logits = ext_logits[0]
            seq_fts = ext_seq_fts[0]

            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if training and self.coef_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    coefs = tf.nn.dropout(coefs, self.coef_drop)
                else:
                    coefs = tf.nn.dropout(coefs, 1.0 - self.coef_drop)

            if training and self.in_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    seq_fts = tf.nn.dropout(seq_fts, self.in_drop)
                else:
                    seq_fts = tf.nn.dropout(seq_fts, 1.0 - self.in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.nn.bias_add(vals,self.bias)

            # residual connection
            if self.residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + self.ret_conv_layer(seq) # activation
                else:
                    ret = ret + seq

            return self.act(ret)  # activation

class sp_attn_head(layers.Layer):
    def __init__(self, output_dim, nb_nodes, in_drop=0.0, coef_drop=0.0,
             residual=False, bias=False, activation=tf.nn.relu, **kwargs):
        super(sp_attn_head, self).__init__(**kwargs)
        self.type = 'sp_attn_head'
        self.act = activation
        self.bias = bias
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.output_dim = output_dim
        self.nb_nodes = nb_nodes

        self.bias = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")

        self.seq_fts_layer = tf.layers.Conv1D(output_dim, kernel_size=1, use_bias=False)
        self.f_1_layer = tf.layers.Conv1D(1, kernel_size=1)
        self.f_2_layer = tf.layers.Conv1D(1, kernel_size=1)
        self.ret_conv_layer = tf.layers.Conv1D(output_dim, kernel_size=1)

# def (seq, att, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    def call(self,inputs, training= None):
        seq, adj_mat = inputs
        with tf.name_scope('sp_attn'):
            if training and self.in_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    seq = tf.nn.dropout(seq, self.in_drop)
                else:
                    seq = tf.nn.dropout(seq, 1.0 - self.in_drop)
                # add a new axis
            ext_seq = tf.expand_dims(seq, axis=0)
            ext_seq_fts = self.seq_fts_layer(ext_seq)


            # simplest self-attention possible
            f_1 = self.f_1_layer(ext_seq_fts)
            f_2 = self.f_2_layer(ext_seq_fts)

            f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
            f_2 = tf.reshape(f_2, (self.nb_nodes, 1))

            f_1 = adj_mat*f_1
            f_2 = adj_mat * tf.transpose(f_2, [1,0])

            logits = tf.sparse_add(f_1, f_2)

            # logits = ext_logits[0]
            seq_fts = tf.squeeze(ext_seq_fts)

            value = tf.nn.leaky_relu(logits.values)

            lrelu = tf.SparseTensor(indices=logits.indices,
                values=value,
                dense_shape=logits.dense_shape)

            coefs = tf.sparse_softmax(lrelu)

            if training and  self.coef_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    coefs = tf.SparseTensor(indices=coefs.indices,
                                            values=tf.nn.dropout(coefs.values, self.coef_drop),
                                            dense_shape=coefs.dense_shape)
                else:
                    coefs = tf.SparseTensor(indices=coefs.indices,
                                            values=tf.nn.dropout(coefs.values, 1.0 - self.coef_drop),
                                            dense_shape=coefs.dense_shape)


            if training and self.in_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    seq_fts = tf.nn.dropout(seq_fts, self.in_drop)
                else:
                    seq_fts = tf.nn.dropout(seq_fts, 1.0 - self.in_drop)


            coefs = tf.sparse_reshape(coefs, [self.nb_nodes, self.nb_nodes])
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
            # vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([self.nb_nodes, self.output_dim])
            ret = tf.nn.bias_add(vals,self.bias)

            # residual connection
            if self.residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + self.ret_conv_layer(seq, ret.shape[-1], 1) # activation
                else:
                    ret = ret + seq

            return self.act(ret)  # activation



class sp_attn_head_l0(layers.Layer):
    def __init__(self, output_dim, nb_nodes, in_drop=0.0, coef_drop=0.0,
             residual=False, bias=False, activation=tf.nn.relu, **kwargs):
        super(sp_attn_head_l0, self).__init__(**kwargs)
        self.type = 'sp_attn_head'
        self.act = activation
        self.bias = bias
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.output_dim = output_dim
        self.nb_nodes = nb_nodes

        self.bias = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")

        self.seq_fts_layer = tf.layers.Conv1D(output_dim, kernel_size=1, use_bias=False)
        self.f_1_layer = tf.layers.Conv1D(1, kernel_size=1)
        self.f_2_layer = tf.layers.Conv1D(1, kernel_size=1)
        self.ret_conv_layer = tf.layers.Conv1D(output_dim, kernel_size=1)

# def (seq, att, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    def call(self, inputs, training= None):
        seq, adj_mat, weight = inputs
        with tf.name_scope('sp_attn'):
            if training and self.in_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    seq = tf.nn.dropout(seq, self.in_drop)
                else:
                    seq = tf.nn.dropout(seq, 1.0 - self.in_drop)
                # add a new axis
            ext_seq = tf.expand_dims(seq, axis=0)
            ext_seq_fts = self.seq_fts_layer(ext_seq)


            # simplest self-attention possible
            f_1 = self.f_1_layer(ext_seq_fts)
            f_2 = self.f_2_layer(ext_seq_fts)

            f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
            f_2 = tf.reshape(f_2, (self.nb_nodes, 1))

            f_1 = adj_mat*f_1
            f_2 = adj_mat* tf.transpose(f_2, [1,0])

            logits = tf.sparse_add(f_1, f_2)
            seq_fts = tf.squeeze(ext_seq_fts)

            value = tf.nn.leaky_relu(logits.values)
            weight = tf.squeeze(weight)
            value = tf.multiply(value,weight)

            lrelu = tf.SparseTensor(indices=logits.indices,
                values=value,
                dense_shape=logits.dense_shape)

            coefs = tf.sparse_softmax(lrelu)

            if training and  self.coef_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    coefs = tf.SparseTensor(indices=coefs.indices,
                                            values=tf.nn.dropout(coefs.values, self.coef_drop),
                                            dense_shape=coefs.dense_shape)
                else:
                    coefs = tf.SparseTensor(indices=coefs.indices,
                                            values=tf.nn.dropout(coefs.values, 1.0 - self.coef_drop),
                                            dense_shape=coefs.dense_shape)


            if training and self.in_drop != 0.0:
                if tf.__version__.startswith('2.'):
                    seq_fts = tf.nn.dropout(seq_fts, self.in_drop)
                else:
                    seq_fts = tf.nn.dropout(seq_fts, 1.0 - self.in_drop)


            coefs = tf.sparse_reshape(coefs, [self.nb_nodes, self.nb_nodes])
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
            # vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([self.nb_nodes, self.output_dim])
            ret = tf.nn.bias_add(vals,self.bias)

            # residual connection
            if self.residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + self.ret_conv_layer(seq, ret.shape[-1], 1) # activation
                else:
                    ret = ret + seq

            return self.act(ret)  # activation
