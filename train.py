from config import args
import tensorflow as tf
import time
from utils import *
from models import GCN
from metrics import *

# Settings

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

# Some preprocessing
features = preprocess_features(features)
support = preprocess_adj(adj)

model = GCN(input_dim=features.shape[1], output_dim=y_train.shape[1])

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

features_tensor = tf.convert_to_tensor(features,dtype=tf.float32)
support_tensor = tf.SparseTensor(*support)
support_tensor = tf.cast(support_tensor,tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train,dtype=tf.float32)
train_mask_tensor = tf.convert_to_tensor(train_mask)
y_test_tensor = tf.convert_to_tensor(y_test,dtype=tf.float32)
test_mask_tensor = tf.convert_to_tensor(test_mask)
y_val_tensor = tf.convert_to_tensor(y_val,dtype=tf.float32)
val_mask_tensor = tf.convert_to_tensor(val_mask)

best_test_acc = 0
best_val_acc = 0
best_val_loss = 10000


curr_step = 0
for epoch in range(args.epochs):

    with tf.GradientTape() as tape:
        output = model.call((features_tensor,support_tensor),training=True)
        cross_loss = masked_softmax_cross_entropy(output, y_train_tensor,train_mask_tensor)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss = cross_loss + args.weight_decay*lossL2
        grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_acc = masked_accuracy(output, y_train_tensor,train_mask_tensor)
    val_acc  = masked_accuracy(output, y_val_tensor,val_mask_tensor)
    val_loss = masked_softmax_cross_entropy(output, y_val_tensor, val_mask_tensor)
    test_acc  = masked_accuracy(output, y_test_tensor,test_mask_tensor)

    if val_acc > best_val_acc:
        curr_step = 0
        best_test_acc = test_acc
        best_val_acc = val_acc
        best_val_loss= val_loss
        # Print results
    else:
        curr_step +=1
    if curr_step > args.early_stop:
        print("Early stopping...")
        break

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cross_loss), "val_loss=",
          "{:.5f}".format(val_loss),
          "train_acc=", "{:.5f}".format(val_acc), "val_acc=", "{:.5f}".format(val_acc),
          "test_acc=", "{:.5f}".format(best_test_acc))

