from __future__ import division
from __future__ import print_function
from config import *
from scipy.sparse import coo_matrix
from supervised_models import SupervisedGraphsage
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler
from utils import *
check_file = './checkpionts/checkpoint'+args.id
from collections import namedtuple

SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

def process_grads(grads):
    if args.dtype=='float64':
        return grads
    grads = [tf.cast(grad, tf.float16) for grad in grads]
    grads = [tf.cast(grad, tf.float32) for grad in grads]
    return grads


def add_noisy_edge(shape, size,nb_noising_edges):
    noise_row = np.random.choice(range(size), nb_noising_edges)
    noise_col = np.random.choice(range(size), nb_noising_edges)
    noise_data = np.ones_like(noise_row)
    noise_adj = coo_matrix((noise_data, (noise_row, noise_col)), shape=shape)
    return noise_adj

def incremental_evaluate(model,minibatch,dataset):
    begin = 0
    leng = len(dataset)
    batch_size = args.val_batch_size
    all_acc = 0
    all_loss = 0
    outputs =[]
    while begin<leng:
        end = begin + batch_size
        if end > leng:
            end = leng
        batch = dataset[begin:end]
        feed_dict = minibatch.batch_feed_dict(batch)
        output,loss,acc = model(feed_dict,training=False)
        all_acc += acc*(end-begin)
        all_loss += loss*(end-begin)
        begin = end
        outputs.append(output)
    output = tf.concat(outputs,axis=0)

    return output, all_loss/leng, all_acc/leng

def main():

    adj, features, labels, idx_train, idx_val, idx_test = load_data_gcn(args.dataset, args.task_type)

    if args.nb_noising_edges > 0:
        noise_adj = add_noisy_edge(adj.shape, labels.shape[0], args.nb_noising_edges)
        adj = noise_adj + adj
        adj[adj > 1] = 1


    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)

    num_classes = len(labels[0])
    # add a row
    features = preprocess_features(features)
    fea = np.asarray(features)
    fea = np.insert(fea, -1, 0, axis=0)
    fea.reshape((features.shape[0] + 1, features.shape[-1]))
    features = fea

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    minibatch = NodeMinibatchIterator(adj, labels, batch_size=args.batch_size, max_degree=args.max_degree,
            train=idx_train, val = idx_val, test = idx_test)


    if args.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(minibatch.adj)

        # two neural layers
        layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                        SAGEInfo("node", sampler, args.samples_2, args.dim_2)]

        model = SupervisedGraphsage(num_classes, features, adj, minibatch.deg, layer_infos)
    elif args.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(minibatch.adj)
        layer_infos = [SAGEInfo("node", sampler, args.samples_1, 2*args.dim_1),
                            SAGEInfo("node", sampler, args.samples_2, 2*args.dim_2)]

        model = SupervisedGraphsage(num_classes, features, adj, minibatch.deg,layer_infos=layer_infos, aggregator_type="gcn", concat=False)

    elif args.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(minibatch.adj)
        layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                            SAGEInfo("node", sampler, args.samples_2, args.dim_2)]

        model = SupervisedGraphsage(num_classes, features, adj, minibatch.deg, layer_infos=layer_infos, aggregator_type="seq")

    elif args.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(minibatch.adj)
        layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                            SAGEInfo("node", sampler, args.samples_2, args.dim_2)]

        model = SupervisedGraphsage(num_classes, features, adj, minibatch.deg, layer_infos=layer_infos,  aggregator_type="maxpool")

    elif args.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(minibatch.adj)
        layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                            SAGEInfo("node", sampler, args.samples_2, args.dim_2)]

        model = SupervisedGraphsage(num_classes, features, adj, minibatch.deg, layer_infos=layer_infos, aggregator_type="meanpool")

    else:
        raise Exception('Error: model name unrecognized.')


    train_feed_dict = minibatch.train_feed_dict

    persist = 0
    best_val_acc = 0
    init_temp = 0.05
    temp = init_temp
    for epoch in range(args.epochs):
        if epoch % args.temp_N == 0:
            decay_temp = np.exp(-1 * args.temp_r * epoch)
            temp = max(0.05, decay_temp)
        train_feed_dict['temperature'] = temp

        with tf.GradientTape() as tape:
            train_output, train_loss, train_acc = model(train_feed_dict,training=True)
        grads = tape.gradient(train_loss, model.trainable_variables)
        grads = process_grads(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_output, val_loss, val_acc = incremental_evaluate(model, minibatch, idx_val)

        if val_acc>best_val_acc:
            best_val_acc = val_acc
            persist = 0
            best_epoch = epoch
            model.save_weights(check_file)
        else:
            persist+=1

        if persist>args.early_stopping:
            break


        if epoch % args.print_every == 0:
            print("epoch:", '%04d' % epoch,
                  "train_loss=", "{:.5f}".format(float(train_loss)),
                  "train_acc=", "{:.5f}".format(float(train_acc)),
                  "val_loss=", "{:.5f}".format(float(val_loss)),
                  "val_acc=", "{:.5f}".format(float(val_acc)))

    print("Optimization Finished!")

    model.load_weights(check_file)
    test_output, test_loss, test_acc = incremental_evaluate(model,minibatch,idx_test)

    print("test_loss=", "{:.5f}".format(float(test_loss)),
          "test_acc=", "{:.5f}".format(float(test_acc)))

if __name__ == '__main__':
    import time
    begin = time.time()
    main()
    end = time.time()
    print('time',end-begin)