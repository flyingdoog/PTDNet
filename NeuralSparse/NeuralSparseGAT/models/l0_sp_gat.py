import tensorflow as tf
from models.base_gattn import BaseGAttN
from config import args, eps, dtype
from layers import sp_attn_head_l0
from scipy.sparse.linalg import svds,eigsh
from scipy.sparse import csc_matrix

class l0SpGAT(BaseGAttN):
    def __init__(self, nb_classes, nb_nodes, n_heads, hid_units, activation=tf.nn.elu,
                 ffd_drop=0.6, attn_drop=0.6, residual=False, feature=None, adj_list=None,
                 **kwargs):
        super(l0SpGAT,self).__init__(**kwargs)

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
            self.attns[0].append(sp_attn_head_l0(output_dim=self.hid_units[0], nb_nodes= nb_nodes,
                                           in_drop=ffd_drop, coef_drop=attn_drop, activation=self.activation,
                                           residual=False))

        for i in range(1, len(self.hid_units)):
            self.attns.append([])
            for _ in range(self.n_heads[i]):
                self.attns[i].append(
                    sp_attn_head_l0(output_dim=self.hid_units[i], nb_nodes= nb_nodes,
                              in_drop=ffd_drop, coef_drop=attn_drop, activation=self.activation, residual=False))

        lid = len(hid_units)
        self.attns.append([])
        for i in range(self.n_heads[-1]):
            self.attns[-1].append(
                sp_attn_head_l0(output_dim=self.nb_classes, nb_nodes= nb_nodes,
                          in_drop=ffd_drop, coef_drop=attn_drop, activation=lambda x: x, residual=False))

        hidden_1 = args.hidden_1
        hidden_2 = args.hidden_2

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##

        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])

        for i in range(len(self.attentions)):
            self.nblayers.append(tf.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))
            self.selflayers.append(tf.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))

            if hidden_2>0:
                self.attentions[i].append(tf.layers.Dense(hidden_2, activation=activation , kernel_initializer = initializer))

            self.attentions[i].append(tf.layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer))

        self.fea_num = feature.shape[1]

        self.attention_layers = []
        self.attention_layers.extend(self.nblayers)
        self.attention_layers.extend(self.selflayers)
        for i in range(len(self.attentions)):
            self.attention_layers.extend(self.attentions[i])


    def set_fea_adj(self,nodes,fea,adj):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat = adj
        self.row = adj.indices[:,0]
        self.col = adj.indices[:,1]

    def get_attention(self, input1, input2, layer=0, training=False):



        nb_layer = self.nblayers[layer]
        selflayer = self.selflayers[layer]
        nn = self.attentions[layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout2
        else:
            dp = 1 - args.dropout2

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        return weight10

    def get_edges(self, input1, input2, layer=1, use_bias=True):
        weight = self.get_attention(input1, input2, layer, use_bias, training=False)
        edges = self.hard_concrete_sample(weight, training=False)
        return edges

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        gamma = 0.0 - args.limit_ab
        zeta = 1.0 + args.limit_ab

        if training:
            debug_var = eps
            gate_inputs_list = []
            L = 1
            for _ in range(L):
                random_noise = tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var,
                                                 dtype=dtype)
                gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
                gate_inputs = tf.sigmoid((gate_inputs + log_alpha) / beta)
                gate_inputs_list.append(gate_inputs)
            gate_inputs = tf.add_n(gate_inputs_list) / float(L)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = tf.clip_by_value(
            stretched_values,
            clip_value_max=1.0,
            clip_value_min=0.0)
        return cliped

    def l0_norm(self, log_alpha, beta):
        gamma = 0 - args.limit_ab
        zeta = 1 + args.limit_ab
        reg_per_weight = tf.sigmoid(log_alpha - beta * tf.cast(tf.math.log(-gamma / zeta), dtype))
        return tf.reduce_mean(reg_per_weight)

    def call(self,inputs, training=None):

        if training:
            lbl_in,msk_in,temperature = inputs
        else:
            lbl_in,msk_in = inputs
            temperature = 1.0

        self.maskes = []
        self.edge_weights = []

        f1_features = tf.gather(self.features, self.row)
        f2_features = tf.gather(self.features, self.col)
        weight = self.get_attention(f1_features, f2_features, layer=0, training=training)
        mask = self.hard_concrete_sample(weight, temperature, training)
        self.edge_weights.append(weight)
        self.maskes.append(mask)

        attns = []
        for i in range(self.n_heads[0]):
            attns.append(self.attns[0][i]((self.features, self.adj_mat,mask), training=training))

        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(self.hid_units)):
            f1_features = tf.gather(self.h_1, self.row)
            f2_features = tf.gather(self.h_1, self.col)
            weight = self.get_attention(f1_features, f2_features, layer=i, training=training)
            mask = self.hard_concrete_sample(weight, temperature, training)
            self.edge_weights.append(weight)
            self.maskes.append(mask)

            attns = []
            for _ in range(self.n_heads[i]):
                attns.append(self.attns[i][_]((h_1,self.adj_mat,mask),training= training))
            h_1 = tf.concat(attns, axis=-1)
        out = []

        for i in range(self.n_heads[-1]):
            f1_features = tf.gather(h_1, self.row)
            f2_features = tf.gather(h_1, self.col)
            weight = self.get_attention(f1_features, f2_features, layer=-1, training=training)
            mask = self.hard_concrete_sample(weight, temperature, training)
            self.edge_weights.append(weight)
            self.maskes.append(mask)
            out.append(self.attns[-1][i]((h_1,self.adj_mat,mask),training= training))

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



        nuclear_loss = tf.zeros([],dtype=dtype)
        l0_loss = tf.zeros([], dtype=dtype)

        # if not training:
        #     for mask in self.maskes:
        #         print(tf.reduce_sum(mask))


        if training and args.lambda1>0.0:
            for weight in self.edge_weights:
                l0_loss += self.l0_norm(weight, temperature)


        if training and args.lambda3>0.0:
            values = []
            for mask in self.maskes:
                mask = tf.squeeze(mask)
                support = tf.SparseTensor(indices=self.adj_mat.indices, values=mask,
                                          dense_shape=self.adj_mat.dense_shape)
                support_dense = tf.sparse.to_dense(support)
                support_trans = tf.transpose(support_dense)
                AA = tf.matmul(support_trans, support_dense)

                if SVD_PI:
                    row_ind = self.adj_mat.indices[:, 0]
                    col_ind = self.adj_mat.indices[:, 1]
                    support_csc = csc_matrix((mask, (row_ind, col_ind)))
                    k = args.k_svd
                    u, s, vh = svds(support_csc, k=k)

                    u = tf.stop_gradient(u)
                    s = tf.stop_gradient(s)
                    vh = tf.stop_gradient(vh)

                    for i in range(k):
                        vi = tf.expand_dims(tf.gather(vh, i), -1)
                        for ite in range(1):
                            vi = tf.matmul(AA, vi)
                            vi_norm = tf.linalg.norm(vi)
                            vi = vi / vi_norm

                        vmv = tf.matmul(tf.transpose(vi), tf.matmul(AA, vi))
                        vv = tf.matmul(tf.transpose(vi), vi)

                        t_vi = tf.math.sqrt(tf.abs(vmv / vv))
                        values.append(t_vi)

                        if k > 1:
                            AA_minus = tf.matmul(AA, tf.matmul(vi, tf.transpose(vi)))
                            AA = AA - AA_minus
                else:
                    trace = tf.linalg.trace(AA)
                    values.append(tf.reduce_sum(trace))

            nuclear_loss = tf.add_n(values)

        return logits, acc, loss, lossL2, l0_loss, nuclear_loss
