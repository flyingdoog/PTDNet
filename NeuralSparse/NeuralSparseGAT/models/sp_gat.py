from config import args
import tensorflow as tf

from layers import sp_attn_head
from models.base_gattn import BaseGAttN

class SpGAT(BaseGAttN):
    def __init__(self, nb_classes, nb_nodes, n_heads, hid_units, activation=tf.nn.elu,
                 ffd_drop=0.6, attn_drop=0.6, residual=False, **kwargs):

        super(SpGAT, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        self.n_heads = n_heads
        self.activation = activation
        self.ffd_drop = ffd_drop
        self.attn_drop = attn_drop
        self.hid_units = hid_units
        self.residual = residual
        self.nb_nodes = nb_nodes

        self.attns = []
        self.attns.append([])
        for _ in range(self.n_heads[0]):
            self.attns[0].append(sp_attn_head(output_dim=self.hid_units[0], nb_nodes= nb_nodes,
                                           in_drop=ffd_drop, coef_drop=attn_drop, activation=self.activation,
                                           residual=False))

        for i in range(1, len(self.hid_units)):
            self.attns.append([])
            for _ in range(self.n_heads[i]):
                self.attns[i].append(
                    sp_attn_head(output_dim=self.hid_units[i], nb_nodes= nb_nodes,
                              in_drop=ffd_drop, coef_drop=attn_drop, activation=self.activation, residual=False))

        lid = len(hid_units)
        self.attns.append([])
        for i in range(self.n_heads[-1]):
            self.attns[-1].append(
                sp_attn_head(output_dim=self.nb_classes, nb_nodes= nb_nodes,
                          in_drop=ffd_drop, coef_drop=attn_drop, activation=lambda x: x, residual=False))

    def call(self,inputs, training=None):

        x, adj_mat, lbl_in, msk_in = inputs

        attns = []
        for i in range(self.n_heads[0]):
            attns.append(self.attns[0][i]((x, adj_mat), training=training))

        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(self.hid_units)):
            attns = []
            for _ in range(self.n_heads[i]):
                attns.append(self.attns[i][_]((h_1,adj_mat), training= training))
            h_1 = tf.concat(attns, axis=-1)
        out = []

        for i in range(self.n_heads[-1]):
            out.append(self.attns[-1][i]((h_1,adj_mat), training= training))
        logits = tf.add_n(out) / self.n_heads[-1]


        log_resh = tf.reshape(logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = self.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        if training:
            vars = self.trainable_variables
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']])
        else:
            lossL2 = 0
        acc = self.masked_accuracy(log_resh, lab_resh, msk_resh)

        return logits, acc, loss, lossL2
