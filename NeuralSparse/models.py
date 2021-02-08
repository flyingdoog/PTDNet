from    layers import *
from    metrics import *
from    config import *

class GumbleGCN(keras.Model):
    def __init__(self, adj_matrix, shape, input_dim, output_dim, k , **kwargs):
        super(GumbleGCN, self).__init__(**kwargs)

        self.adj_matrix = adj_matrix
        self.shape = shape
        self.input_dim = input_dim  # 1433
        self.output_dim = output_dim
        self.k = k


        layer1 = GraphConvolution(input_dim=self.input_dim, # 1433
                                            output_dim=params['hidden1'], # 16
                                            activation=tf.nn.relu,
                                            dropout=args.dropout,
                                            is_sparse_inputs=False)

        layer2  = GraphConvolution(input_dim=params['hidden1'], # 16
                                            output_dim=params['hidden2'],
                                            activation=tf.nn.relu,
                                            dropout=args.dropout)

        layer3  = Dense(input_dim=params['hidden2'], # 16
                                            output_dim=self.output_dim, # 7
                                            activation=lambda x: x,
                                            dropout=args.dropout)


        self.layers_ = []
        self.layers_.append(layer1)
        self.layers_.append(layer2)
        self.layers_.append(layer3)

        self.slayer1 = tf.keras.layers.Dense(32, tf.nn.relu, name='f12dense1')
        self.slayer2 = tf.keras.layers.Dense(1, use_bias=True, name='f12dense2')
        self.sparse_layers = []
        self.sparse_layers.append(self.slayer1)
        self.sparse_layers.append(self.slayer2)
    def sample_gumbel(self, shape):
        """Sample from Gumbel(0, 1)"""
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, istrain):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        r = self.sample_gumbel(tf.shape(logits.values))
        values = tf.cond(istrain is not None, lambda: tf.math.log(logits.values) + r, lambda: tf.math.log(logits.values))
        values /= temperature
        y = tf.SparseTensor(self.adj_matrix.indices, values, self.shape)
        return tf.sparse.softmax(y)


    def call(self, inputs, training=None):
        """
        :param inputs:
        :param training:
        :return:
        """
        x, label, mask, temperature = inputs

        f1 = tf.gather(x, self.adj_matrix.indices[:, 0])
        f2 = tf.gather(x, self.adj_matrix.indices[:, 1])
        auv = tf.expand_dims(self.adj_matrix.values, -1)
        temp = tf.concat([f1, f2, auv], -1)

        temp = self.slayer1(temp)
        temp = self.slayer2(temp)
        z = tf.reshape(temp, [-1])

        z_matrix= tf.SparseTensor(self.adj_matrix.indices, z, self.shape)
        pi = tf.sparse.softmax(z_matrix)

        y = self.gumbel_softmax_sample(pi, temperature, training)
        y_dense = tf.sparse.to_dense(y)

        top_k_v, top_k_i = tf.math.top_k(y_dense,self.k)

        kth = tf.reduce_min(top_k_v,-1)+eps # N,
        kth = tf.expand_dims(kth,-1)
        kth = tf.tile(kth,[1,kth.shape[0]]) # N,N
        mask2 = tf.greater_equal(y_dense, kth)
        mask2 = tf.cast(mask2, tf.float32)
        row_sum = tf.reduce_sum(mask2,-1)

        dense_support = mask2

        if args.weighted:
            dense_support = tf.multiply(y_dense,mask2)
        else:
            print('no gradient bug here!')
            exit()
        # norm
        self_edge = tf.eye(self.shape[0],self.shape[0])
        dense_support = dense_support+self_edge

        rowsum = tf.reduce_sum(dense_support,-1)+1e-6 # to avoid NaN

        d_inv_sqrt = tf.reshape(tf.math.pow(rowsum, -0.5),[-1])  # D^-0.5
        d_mat_inv_sqer = tf.linalg.diag(d_inv_sqrt)
        ad = tf.matmul(dense_support,d_mat_inv_sqer)
        adt = tf.transpose(ad,name='adt')
        dadt=tf.matmul(adt,d_mat_inv_sqer)
        suport = dadt

        hidden = self.layers_[0]((x, suport), training)

        hidden = self.layers_[1]((hidden, suport), training)
        output = self.layers_[2](hidden, training)


        # # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += params['weight_decay'] * tf.nn.l2_loss(var)

        # Cross entropy error
        loss += masked_softmax_cross_entropy(output, label, mask)

        acc = masked_accuracy(output, label, mask)

        return loss, acc
