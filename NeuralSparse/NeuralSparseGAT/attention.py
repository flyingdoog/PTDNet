import tensorflow as tf
import numpy as np
from layers import Dense
from config import params as global_params
class Attention(object):
    def __init__(self, placeholders,params=None,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.limit_a = params['limit_a']
        self.limit_b = params['limit_b']
        self.epsilon = 1e-7
        self.adj_matrix_tf = placeholders['adj']
        self.adj_matrix = params['adj']
        self.node_size = self.adj_matrix.shape[0]

        self.istrain = placeholders['istrain']
        self.params = params
        self.edge_size = params['nnz']

        self.lambda1 = placeholders['lambda1']
        self.lambda2 = params['lambda2']
        self.degree = params['degree']
        self.indices = np.vstack((self.adj_matrix.row, self.adj_matrix.col)).transpose()
        self.shape = self.adj_matrix.shape

        self.temperature = placeholders['temperature']
        self.hard = params['hard']
        self.log_alpha_initializer = tf.random_normal_initializer(mean=global_params['init_values'], stddev=0.01)#
        self.log_alpha = tf.get_variable("att_log_alpha",shape=self.edge_size,initializer=self.log_alpha_initializer,dtype=tf.float32,trainable=True)


        self.inputs = placeholders['features']
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=params['weight_decay'])

        self.build()

    def build(self):
        f1 = tf.gather(self.inputs, self.adj_matrix_tf.indices[:, 0])
        f2 = tf.gather(self.inputs, self.adj_matrix_tf.indices[:, 1])

        f1layer = Dense(f1.shape[-1],16,name='att_f1layer',act = tf.nn.leaky_relu)
        f2layer = Dense(f1.shape[-1],16,name='att_f2layer',act = tf.nn.leaky_relu)

        f1 = f1layer(f1)
        f2 = f2layer(f2)
        temp = tf.concat((f1, f2), -1)

        f12layer1 = Dense(temp.shape[-1],16,name='att_f12layer1',act = tf.nn.leaky_relu)
        temp = f12layer1(temp)

        f12layer2 = Dense(temp.shape[-1],1,name='att_f12layer2',act = lambda x:x,bias=False)
        temp = f12layer2(temp)

        logits = tf.reshape(temp, [-1])
        logits = tf.nn.tanh(logits)

        self.log_alpha = logits + self.log_alpha


        self.mask_values = self.hard_concrete_sample()

        if self.hard:
            hard_mask_values = tf.clip_by_value(self.mask_values, clip_value_max=self.epsilon, clip_value_min=0.0)
            hard_mask_values *= 1.0 / self.epsilon  # (0,1)
            self.mask_values = tf.stop_gradient(hard_mask_values - self.mask_values) + self.mask_values

        self.sp_support = tf.SparseTensor(self.indices, self.mask_values, self.shape)

        self._loss()

    def hard_concrete_sample(self):
        """Uniform random numbers for the concrete distribution"""
        log_alpha = self.log_alpha
        beta = self.temperature
        gamma = self.limit_a
        zeta = self.limit_b
        eps = self.epsilon
        random_noise = tf.random_uniform(tf.shape(self.log_alpha), minval=0.0, maxval=1.0)
        gate_inputs = tf.log(random_noise + eps) - tf.log(1.0 - random_noise)
        gate_inputs *= self.istrain # 0/1  when is not train beta = 1, thus, hard_concrete_smaple = hard_concrete_mean
        gate_inputs = tf.sigmoid((gate_inputs + log_alpha) / beta)
        stretched_values = gate_inputs * (zeta - gamma) + gamma

        return tf.clip_by_value(
            stretched_values,
            clip_value_max=1.0,
            clip_value_min=0.0)

    def l0_norm(self):
        log_alpha = self.log_alpha
        beta = self.temperature
        gamma = self.limit_a
        zeta = self.limit_b

        # Value of the CDF of the hard-concrete distribution evaluated at 0
        reg_per_weight = tf.sigmoid(log_alpha - beta * tf.log(-gamma / zeta))
        return tf.reduce_mean(reg_per_weight)

    def _loss(self):
        # Weight decay loss
        # self.loss = tf.losses.get_regularization_loss()
        self.loss = 0
        # Cross entropy error
        self.l0_reg = self.l0_norm()

        self.loss += self.lambda1*self.l0_reg

        self.kl_loss = 0
        # masked_degree = tf.reduce_sum(self.support,axis=-1)
        #
        # masked_degree = masked_degree/(tf.reduce_sum(masked_degree))+1e-6
        # self.kl_loss = tf.reduce_sum(self.degree * tf.log(self.degree / masked_degree))

        self.loss += self.lambda2*self.kl_loss