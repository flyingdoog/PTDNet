import  argparse
import  os
import numpy as np
import random

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--multi_label',type=bool, default=False)
args.add_argument('--model', default='l0gcn')
args.add_argument('--learning_rate',type=float, default=0.001) #0.01 for adam
args.add_argument('--epochs',type=int, default=3000)
args.add_argument('--hidden1', type=int,default=128)
args.add_argument('--hidden2', default=64)
args.add_argument('--dropout',type=float, default=0.0)
args.add_argument('--dropout2', type=float,default=0.0)

args.add_argument('--weight_decay',type=float, default=0.0)
args.add_argument('--early_stopping',type=int, default=100)

args.add_argument('--task_type', default='semi')

args.add_argument('--init_value', type=float, default=1.0)  #
args.add_argument('--use_bias', type=bool, default=False)  #
args.add_argument('--seed', type= int, default=1234)
args.add_argument('--whole_batch', type= bool, default=True)

args.add_argument('--topK', type=int, default=5, help='the number of samples')
args.add_argument('--weighted', type= bool, default=True)

args.add_argument('--hidden_1', type=int, default=32)
args.add_argument('--hidden_2', type=int, default=0)
args.add_argument('--act', default='leaky_relu')
args.add_argument('--initializer', default='glorot')

args.add_argument('--temp_r',type=float, default=1e-3)
args.add_argument('--temp_N',type=int, default=50)


args.add_argument('--trails', type=int, default=1)
args.add_argument('--print_every', type=int, default=1)
args.add_argument('--use_gpu', type=str, default="0")  #
args.add_argument('--device', type=int, default=0)  #
args.add_argument('--setting', type=str, default="test only")  #
args.add_argument('--id', type=str, default="hello test only")  #

args.add_argument('--dtype', type=str, default='float32')  #



# Exp2
args.add_argument('--nb_noising_edges', type=int, default=0, help='noisy edges')

args = args.parse_args()
print(args)
params = vars(args)
need_check_edge = False
val_whole_batch = True

devices = ['0','1']

real_device = args.device%len(devices)
os.environ["CUDA_VISIBLE_DEVICES"] = devices[real_device]
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


import tensorflow as tf

if tf.__version__.startswith('1.'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    # ster = tf.contrib.summary.create_file_writer("log")ummary_wri

seed = args.seed
np.random.seed(seed)
if tf.__version__.startswith('2.'):
    tf.random.set_seed(seed)
else:
    tf.set_random_seed(seed)
random.seed(args.seed)
eps = 1e-8

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

SVD_PI = True
lowpre = True
if args.dataset =='ppi':
    args.multi_label = True
else:
    args.multi_label = False
