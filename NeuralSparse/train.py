from    utils import *
from    models import GumbleGCN
from    config import args
from scipy.sparse import coo_matrix
import  os


def add_noisy_edge(shape, size,nb_noising_edges):
    noise_row = np.random.choice(range(size), nb_noising_edges)
    noise_col = np.random.choice(range(size), nb_noising_edges)
    noise_data = np.ones_like(noise_row)
    noise_adj = coo_matrix((noise_data, (noise_row, noise_col)), shape=shape)
    return noise_adj


if __name__ == '__main__':

    # load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    y_train = labels
    y_val = labels
    y_test = labels

    if args.nb_noising_edges > 0:
        noise_adj = add_noisy_edge(adj.shape, labels.shape[0], args.nb_noising_edges)
        adj = noise_adj + adj
        adj[adj > 1] = 1


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    features = preprocess_features(features)
    print('features coordinates::', features[0].shape)
    print('features data::', features[1].shape)
    print('features shape::', features[2])

    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()

    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    shape = adj.shape

    adj = (indices, adj.data, adj.shape)

    adj = tf.cast(tf.SparseTensor(*adj), dtype=tf.float32)



    model = GumbleGCN(adj, shape = shape, input_dim=features.shape[-1], output_dim=labels.shape[-1], k=args.topK)

    train_label = tf.convert_to_tensor(y_train)
    train_mask = tf.convert_to_tensor(train_mask)
    val_label = tf.convert_to_tensor(y_val)
    val_mask = tf.convert_to_tensor(val_mask)
    test_label = tf.convert_to_tensor(y_test)
    test_mask = tf.convert_to_tensor(test_mask)
    features = tf.convert_to_tensor(features)
    dropout = args.dropout
    feature_tensor = tf.convert_to_tensor(features)

    if tf.__version__.startswith('2.'):
        optimizer = tf.optimizers.Adam(lr=args.learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    persist = 0
    best_val_acc = 0
    epochs = args.epochs
    init_temp = 0.05
    for epoch in range(epochs):
        if epoch%args.temp_N==0:
            decay_temp = np.exp(-1*args.temp_r*epoch)
            temp = max(0.05,decay_temp)
        with tf.GradientTape() as tape:
            loss, acc = model((features, train_label, train_mask, temp),training=True)
        # debug
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_loss, val_acc = model((features, val_label, val_mask, 1.0), training=False)
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            persist = 0
            model.save_weights('easy_checkpoint')
        else:
            persist+=1
        if persist>args.early_stopping:
            break
        if epoch % 1 == 0:
            print(epoch,'temp',temp,'loss',float(loss), float(acc), '\tval:', float(val_acc))


    model.load_weights('easy_checkpoint')
    test_loss, test_acc = model((features, test_label, test_mask, 1.0), training=False)


    print('\ttest:', float(test_loss), float(test_acc))
