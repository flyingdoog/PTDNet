from config import *
from scipy.sparse import coo_matrix
from models import NeuralSparseGAT
from utils import process
from myutils import *

checkpt_file = 'pre_trained/'+args.dataset+'/mod_sp_'+args.dataset+args.id+'.ckpt'
import tensorflow as tf
dataset = args.dataset
nb_epochs = args.epochs
patience = args.early_stop
lr = args.learning_rate  # learning rate
l2_coef = args.l2_coef  # weight decay
hid_units = [args.hid_units]
n_heads = []
for head in args.n_heads.split('-'):
    n_heads.append(int(head))

residual = args.residual

ffd_drop=args.dropout
attn_drop=args.dropout

nonlinearity = tf.nn.elu

if args.act == 'relu':
    nonlinearity = tf.nn.relu
elif args.act == 'elu':
    nonlinearity = tf.nn.elu
if args.act == 'leaky_relu':
    nonlinearity = tf.nn.leaky_relu

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))

ori_adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset, params['task_type'])
features, spars = process.preprocess_features(features)
ori_features = features

nb_noising_edges = params['nb_noising_edges']
if nb_noising_edges>0:
    noise_row = np.random.choice(range(len(train_mask)), nb_noising_edges)
    noise_col = np.random.choice(range(len(train_mask)), nb_noising_edges)
    noise_data = np.ones_like(noise_row)
    noise_adj = coo_matrix((noise_data,(noise_row,noise_col)),shape=ori_adj.shape)
    ori_adj = ori_adj+noise_adj
    ori_adj[ori_adj>1]=1
adj = ori_adj


nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

biases, norm_adj = process.mypreprocess_adj_bias_1(adj)


features_tensor = tf.convert_to_tensor(features,dtype=tf.float32)
biases_tensor = tf.SparseTensor(*biases)

y_train_tensor = tf.convert_to_tensor(y_train,dtype=tf.float32)
train_mask_tensor = tf.convert_to_tensor(train_mask)
y_test_tensor = tf.convert_to_tensor(y_test,dtype=tf.float32)
test_mask_tensor = tf.convert_to_tensor(test_mask)
y_val_tensor = tf.convert_to_tensor(y_val,dtype=tf.float32)
val_mask_tensor = tf.convert_to_tensor(val_mask)

model = NeuralSparseGAT(nb_classes=nb_classes, nb_nodes = features.shape[0], n_heads=n_heads, hid_units=hid_units, activation=nonlinearity,
                 ffd_drop=ffd_drop, attn_drop=attn_drop, residual=False, feature=features, adj_list=biases)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

model.set_fea_adj(np.array(range(nb_nodes)),features_tensor,biases_tensor)

vacc_mx = 0
vlss_mn = np.inf
best_test_acc = 0.0

init_temp = 0.05
temp = init_temp
for epoch in range(args.epochs):
    if epoch % args.temp_N == 0:
        decay_temp = np.exp(-1 * args.temp_r * epoch)
        temperature = max(0.05, decay_temp)
    with tf.GradientTape() as tape:
        logits, train_acc, loss, l2loss = model.call((y_train_tensor, train_mask_tensor,temperature), training=True)
        train_loss = loss + l2_coef*l2loss
        grads = tape.gradient(train_loss, model.trainable_variables)
        grads = process_grads(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    val_output, val_acc,val_loss,_ = model.call((y_val_tensor,val_mask_tensor),training=False)

    print('epoch=%d, Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f, test_acc=%.5f' %
            (epoch,loss, train_acc, val_loss, val_acc,best_test_acc))

    if val_acc >= vacc_mx:
        vacc_early_model = val_acc
        vlss_early_model = val_loss
        test_output,test_acc,ts_loss,_  = model.call((y_test_tensor, test_mask_tensor),  training=False)
        best_test_acc = test_acc.numpy()
        vacc_mx = np.max((val_acc, vacc_mx))
        vlss_mn = np.min((val_loss, vlss_mn))
        curr_step = 0
    else:
        curr_step += 1
        if curr_step == patience:
            print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
            print('Early stop model validation loss: ', vlss_early_model.numpy(), ', accuracy: ', vacc_early_model.numpy())
            break

    train_loss_avg = 0
    train_acc_avg = 0
    val_loss_avg = 0
    val_acc_avg = 0

# model.load_weights(checkpt_file)


print('; Test accuracy:', best_test_acc)


# record_db(best_test_acc)
# remove_device()

