
from    config import *
import datetime
import numpy as np
import  tensorflow as tf
import time
import logging
from data_loader import data_loader


from    utils import *
from    models import GCN, l0GCN
from    metrics import *

whole_batch = args.whole_batch
print('tf version:', tf.__version__)
check_file = './checkpionts/checkpoint'+args.id

logging.getLogger('tensorflow').disabled = True

def process_grads(grads):
    if args.dtype=='float64':
        return grads
    grads = [tf.cast(grad, tf.float16) for grad in grads]
    grads = [tf.cast(grad, tf.float32) for grad in grads]
    return grads

def main():

    adj, features, all_labels, train, val, test = load_data(args.dataset, task_type=args.task_type)
    # adj, features, all_labels, train, val, test = load_syn_data()


    MAX_NBS = params['max_degree']
    train = np.array(train)
    val = np.array(val)
    test = np.array(test)

    print('adj:', adj.shape)
    print('features:', features.shape)

    # padding adj to N*K, where K is the number of nbs
    adj_list = get_adj_list(adj, MAX_NBS)

    #gather -1 is not supported by cpu version of tensorflow.
    cpu_adj_list = adj_list.copy()
    cpu_adj_list[adj_list==-1]=adj.shape[0]-1

    feed_adj_list = cpu_adj_list

    dl = data_loader(features, train, val, test)


    adj_mask = adj_list + 1
    adj_mask[adj_mask > 0] = 1
    signed_adj_mask = get_signed_adj_mask(adj_list, all_labels)
    signed_adj_mask = np.multiply(adj_mask,signed_adj_mask)
    all_edges = np.sum(adj_mask)
    pos_edges_neg_edges = np.sum(signed_adj_mask)
    pos_edges = (all_edges+pos_edges_neg_edges)/2
    neg_edges = all_edges-pos_edges
    print(pos_edges,neg_edges)

    rows = np.expand_dims(np.array(list(range(adj.shape[0]))),-1)
    rows_tile = np.tile(rows,(1,MAX_NBS))

    features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
    print('features coordinates::', features[0].shape)
    print('features data::', features[1].shape)
    print('features shape::', features[2])

    # add a row
    fea = np.asarray(features)
    fea = np.insert(fea, -1, 0, axis=0)
    fea.reshape((features.shape[0] + 1, features.shape[-1]))
    features = fea

    test_accs = []

    def check_edges(batch, edges=None):
        row, col, data = get_edges(adj_list, rows_tile, signed_adj_mask, adj_mask, batch)
        if edges is None:
            row, col, data = get_edges(adj_list, rows_tile, signed_adj_mask, adj_mask, batch)
            row_feature = features[row]
            col_feature = features[col]
            edges = model.get_edges(row_feature, col_feature, layer=1, use_bias=True).numpy().flatten()
        else:
            edges = np.extract(adj_mask[batch],edges).flatten()

        all_sum = np.sum(edges)
        pos_neg_sum = np.sum(edges*data)
        pos = (all_sum+pos_neg_sum)/2
        neg = all_sum-pos

        all_edges = len(row)
        pos_edges_neg_edges = np.sum(data)
        pos_edges = (all_edges + pos_edges_neg_edges) / 2
        neg_edges = all_edges - pos_edges

        cross_entropy = 0#log_loss(data, edges)
        return cross_entropy, (pos/pos_edges), (neg/neg_edges)

    t_begin = time.time()
    lambda3 = args.lambda3

    if lambda3 > 0:
        train_mask = adj_mask[train]

    if args.model == 'gcn':
        model = GCN(input_dim=features.shape[1], output_dim=all_labels.shape[1], num_features_nonzero=features[1].shape, \
                    feature=features, label=all_labels, adj_list=feed_adj_list, adj_mask=adj_mask, multi_label = args.multi_label)  # [1433]

    elif args.model == 'l0gcn':
        model = l0GCN(input_dim=features.shape[1], output_dim=all_labels.shape[1],
                      num_features_nonzero=features[1].shape, \
                      feature=features, label=all_labels, adj_list=feed_adj_list, rows_tile=rows_tile,\
                      adj_mask=adj_mask, signed_adj_mask= signed_adj_mask, multi_label = args.multi_label)

    model.load_weights(check_file)
    print('read checkpoint done')

    if whole_batch:
        if args.model == 'l0gcn':
            test_output, test_loss, test_acc, test_l0mask1, test_l0mask2 = model((test, len(test)))
        else:
            test_output, test_loss, test_acc = model(test)
    else:
        test_outputs = []
        while not dl.test_end():
            batch = dl.get_test_batch(batch_size=args.val_batch_size)
            if args.model == 'l0gcn':
                test_output, test_loss, test_acc, test_l0mask1, test_l0mask2 = model((batch, len(batch)))
            else:
                test_output, test_loss, test_acc = model(batch)
            test_outputs.append(test_output)
        test_output = np.concatenate(test_outputs)

    if args.multi_label:
        if not isinstance(test_output, np.ndarray):
            test_output = test_output.numpy()
        test_acc = calc_f1(test_output, all_labels[test])

    test_accs.append(test_acc)

    if args.model=='l0gcn'  and args.whole_batch:
        mask, l0mask = model.get_maskes()
        model((train, len(train)))
        train_mask, train_l0mask = model.get_maskes()
        model((val, len(val)))
        val_mask, val_l0mask = model.get_maskes()
        print('train edges', np.sum(train_mask), np.sum(train_l0mask))
        print('val edges', np.sum(val_mask), np.sum(val_l0mask))
    print('\ttest:',float(test_acc))


def record_db(mean):
    sql = 'INSERT INTO dgx1.GCN VALUES (\''+args.dataset+'\',\''+args.id+'\','+str(mean)+',\''+args.setting+'\')'
    print(sql)
    mycursor.execute(sql)
    mydb.commit()

def remove_device():
    cmd = "delete FROM dgx1.device where device="+str(args.device)
    print(cmd)
    mycursor.execute(cmd)
    mydb.commit()


def get_edges(all_col,all_row,all_data,all_mask,batch):
    col = all_col[batch]
    row = all_row[batch]
    data = all_data[batch]
    mask = all_mask[batch]

    rows_non_empty = np.extract(mask, row).flatten()
    col_non_empty = np.extract(mask, col).flatten()
    data_non_empty = np.extract(mask, data).flatten()

    return rows_non_empty, col_non_empty, data_non_empty


if __name__ == '__main__':
    begin = time.time()
    main()
    end = time.time()
    print('time',end-begin)

