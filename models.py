from config import *
from layers import *
from metrics import *
from tensorflow import keras
from scipy.sparse.linalg import svds,eigsh
from scipy.sparse import csc_matrix

class GCN(keras.Model):
    def __init__(self, input_dim, output_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        self.layers_ = []

        layer0 = GraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0],
                                  activation=tf.nn.relu)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_],
                                      activation=tf.nn.relu)
            self.layers_.append(layertemp)

        layer_1 = GraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim,
                                            activation=lambda x: x)
        self.layers_.append(layer_1)
        self.hiddens = hiddens

    def call(self,inputs,training=None):
        x, support  = inputs
        for layer in self.layers_:
            x = layer.call((x,support),training)
        return x

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])





class GCN_dropedge(GCN):
    def __init__(self, input_dim, output_dim,adj, **kwargs):
        super(GCN_dropedge, self).__init__(input_dim,output_dim,**kwargs)
        self.adj = adj
        self.nodesize = adj.shape[0]

        rowsum = tf.sparse.reduce_sum(adj, axis=-1) + 1e-6
        d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
        row_inv_sqrt = tf.gather(d_inv_sqrt, adj.indices[:, 0])
        col_inv_sqrt = tf.gather(d_inv_sqrt, adj.indices[:, 1])
        values = tf.multiply(adj.values, row_inv_sqrt)
        values = tf.multiply(values, col_inv_sqrt)

        self.support = tf.SparseTensor(indices=adj.indices,
                                  values=values,
                                  dense_shape=adj.shape)

        if tf.__version__.startswith('2.'):
            self.dp = args.dropout
        else:
            self.dp = 1 - args.dropout


    def call(self,inputs,training=None):

        x  = inputs
        if training:
            indices = tf.cast(self.adj.indices,tf.float32)
            data = tf.expand_dims(self.adj.values,-1)
            indices_data = tf.concat([indices,data],axis=-1)
            mask = tf.ones([indices.shape[0]])
            mask = tf.nn.dropout(mask,self.dp) 
            mask = tf.clip_by_value(mask, clip_value_max=1.0, clip_value_min=0.0)
            edges = tf.reduce_sum(mask)
            mask = tf.cast(mask,bool)
            dropedge_indices_data = tf.boolean_mask(indices_data,mask)
            dropedge_indices = tf.cast(dropedge_indices_data[:,:2],tf.int64)
            dropedge_values = dropedge_indices_data[:,2]

            dropedge_adj = tf.SparseTensor(indices=dropedge_indices,
                                      values=dropedge_values,
                                      dense_shape=self.adj.shape)
            dropedge_adj = tf.sparse.add(dropedge_adj,tf.sparse.eye(self.nodesize,dtype=dtype))
            rowsum = tf.sparse.reduce_sum(dropedge_adj, axis=-1)
            d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
            row_inv_sqrt = tf.gather(d_inv_sqrt, dropedge_adj.indices[:, 0])
            col_inv_sqrt = tf.gather(d_inv_sqrt, dropedge_adj.indices[:,1])
            values = tf.multiply(dropedge_adj.values, row_inv_sqrt)
            values = tf.multiply(values, col_inv_sqrt)

            support = tf.SparseTensor(indices=dropedge_adj.indices,
                                      values=values,
                                      dense_shape=dropedge_adj.shape)
        else:
            support = self.support

        for layer in self.layers_:
            x = layer.call((x,support),training)
        return x

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])


class MLP(GCN):
    def __init__(self, input_dim, output_dim,adj, **kwargs):
        super(MLP, self).__init__(input_dim,output_dim,**kwargs)
        self.adj = adj
        self.nodesize = adj.shape[0]
        self.support = tf.sparse.eye(self.nodesize,dtype=dtype)

    def call(self,inputs,training=None):

        x  = inputs

        for layer in self.layers_:
            x = layer.call((x,self.support),training)
        return x

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])



class PTDNetGCN(GCN):
    def __init__(self, input_dim, output_dim,activation=tf.nn.relu, **kwargs):
        super(PTDNetGCN, self).__init__(input_dim,output_dim,**kwargs)

        hidden_1 = args.denoise_hidden_1
        hidden_2 = args.denoise_hidden_2
        self.edge_weights = []
        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##

        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        for hidden in self.hiddens:
            self.attentions.append([])

        for i in range(len(self.attentions)):
            self.nblayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))
            self.selflayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))

            if hidden_2>0:
                self.attentions[i].append(tf.keras.layers.Dense(hidden_2, activation=activation , kernel_initializer = initializer))

            self.attentions[i].append(tf.keras.layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer))

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
            dp = args.dropout
        else:
            dp = 1 - args.dropout

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
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias+tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var, dtype=dtype)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = tf.clip_by_value(
            stretched_values,
            clip_value_max=1.0,
            clip_value_min=0.0)
        return cliped

    def l0_norm(self, log_alpha, beta):
        gamma = args.gamma
        zeta = args.zeta
        reg_per_weight = tf.sigmoid(log_alpha - beta * tf.cast(tf.math.log(-gamma / zeta), dtype))
        return tf.reduce_mean(reg_per_weight)

    def call(self,inputs, training=None):

        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.edge_maskes = []

        self.maskes = []


        x = self.features
        layer_index = 0
        for layer in self.layers_:
            xs = []
            for l in range(args.L):
                f1_features = tf.gather(x, self.row)
                f2_features = tf.gather(x, self.col)
                weight = self.get_attention(f1_features, f2_features, layer=layer_index, training=training)
                mask = self.hard_concrete_sample(weight, temperature, training)
                mask_sum = tf.reduce_sum(mask)
                self.edge_weights.append(weight)
                self.maskes.append(mask)
                mask = tf.squeeze(mask)
                adj = tf.SparseTensor(indices=self.adj_mat.indices,
                                      values=mask,
                                      dense_shape=self.adj_mat.shape)
                # norm
                adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype=dtype))

                row = adj.indices[:, 0]
                col = adj.indices[:, 1]

                rowsum = tf.sparse.reduce_sum(adj, axis=-1)#+1e-6
                d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
                d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
                row_inv_sqrt = tf.gather(d_inv_sqrt,row)
                col_inv_sqrt = tf.gather(d_inv_sqrt,col)
                values = tf.multiply(adj.values,row_inv_sqrt)
                values = tf.multiply(values,col_inv_sqrt)

                support = tf.SparseTensor(indices=adj.indices,
                                      values=values,
                                      dense_shape=adj.shape)
                nextx = layer.call((x,support),training)
                xs.append(nextx)
            x = tf.reduce_mean(xs,0)
            layer_index +=1
        return x

    def lossl0(self,temperature):
        l0_loss = tf.zeros([], dtype=dtype)
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def nuclear(self):

        nuclear_loss = tf.zeros([],dtype=dtype)
        values = []
        if args.lambda3==0:
            return 0
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
                support_csc = csc_matrix((mask.numpy(), (row_ind.numpy(), col_ind.numpy())))
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

        return nuclear_loss
