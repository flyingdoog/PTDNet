import tensorflow as tf

from layers import Dense
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from config import *
from attention import Attention
from  tensorflow import keras

class SupervisedGraphsage(keras.Model):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", **kwargs):
        '''
        Args:
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
        '''
        super(SupervisedGraphsage, self).__init__(**kwargs)

        self.adj_info = adj
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)

        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.dims = [features.shape[1]]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        dim_mult = 2 if self.concat else 1

        self.node_pred = Dense(dim_mult*self.dims[-1], self.num_classes,
                dropout=args.dropout, act=lambda x : x)


        self.num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        self.aggregators = self.aggregate(aggregator_type,self.dims,concat=self.concat)
        self.layers_ = []
        self.layers_.extend(self.aggregators)

        for p in self.trainable_variables:
            print(p.name, p.shape)

    def call(self, inputs, training=True):

        batch_size = inputs['batch_size']
        batch = inputs['batch']
        labels = inputs['labels']
        if training:
            temperature = inputs['temperature']
        else:
            temperature = 1.0

        samples, support_sizes = self.sample(batch, self.layer_infos, batch_size=batch_size)

        hidden = []
        for hidden_index in range(0,len(samples)):
            ids = tf.cast(samples[hidden_index],tf.int32)
            fea = tf.nn.embedding_lookup(self.features, ids)
            hidden.append(fea)
        edge_weights = []
        for layer in range(len(self.num_samples)):
            aggregator = self.aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(self.num_samples) - layer):
                dim_mult = 2 if self.concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              self.num_samples[len(self.num_samples) - hop - 1],
                              dim_mult*self.dims[layer]]
                h,edge_weight = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_dims),temperature),training)
                next_hidden.append(h)
                edge_weights.append(edge_weight)
            hidden = next_hidden
        outputs1 = hidden[0]

        outputs1 = tf.nn.l2_normalize(outputs1, 1)
        self.node_preds = self.node_pred(outputs1)

        loss = self._loss(labels, edge_weights, temperature,samples, training=training)
        preds = self.predict()
        acc = self.cal_accuracy(preds,labels)
        return preds, loss, acc

    def cal_accuracy(self,preds, labels):
        """
        Accuracy with masking.
        """
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

    def _loss(self,labels, edge_weights,temperature,samples, training=True):

        l2loss = tf.zeros([],dtype=dtype)
        for layer in self.aggregators:
            for var in layer.trainable_variables:
                l2loss += tf.nn.l2_loss(var)

        for layer in self.attentions:
            for var in layer.trainable_variables:
                l2loss += tf.nn.l2_loss(var)

        # classification loss
        cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.node_preds,labels=labels))


        loss = cross_loss+args.weight_decay*l2loss
        return loss

    def predict(self):
        return tf.nn.softmax(self.node_preds)

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """

        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        atts = []
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size, ]))
            support_sizes.append(support_size)
        return samples, support_sizes


    def aggregate(self,aggregator_type, dims, name=None, concat=False):


        if aggregator_type == "mean":
            aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        self.attentions = []
        for _ in range(len(self.layer_infos)):
            self.attentions.append(Attention())

        aggregators = []
        for layer in range(len(self.num_samples)):
            dim_mult = 2 if concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == len(self.num_samples) - 1:
                aggregator = aggregator_cls(dim_mult*dims[layer], dims[layer+1], activation=lambda x : x,
                        dropout=args.dropout, attention=self.attentions[layer],
                        name=name, concat=concat)
            else:
                aggregator = aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                        dropout=args.dropout,attention=self.attentions[layer],
                        name=name, concat=concat)
            aggregators.append(aggregator)

        return aggregators
