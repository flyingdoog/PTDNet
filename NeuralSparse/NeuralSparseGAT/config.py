import argparse
import numpy as np
import os
import random


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cora', help="Dataset string")# 'cora', 'citeseer', 'pubmed'
    parser.add_argument('--id', type=str, default='default_id', help='activation funciton')  #
    parser.add_argument('--device', type=int, default=0)  #
    parser.add_argument('--setting', type=str, default="test only")  #
    parser.add_argument('--machine', type=str, default='local')


    parser.add_argument("--learning_rate", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train.')
    parser.add_argument('--task_type', type=str, default='semi')
    parser.add_argument('--dropout',type=float, default=0., help='dropout rate (1 - keep probability).')
    parser.add_argument('--l2_coef',type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--early_stop', type=int, default= 100, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #

    parser.add_argument('--hid_units', type=int, default=32, help='numbers of hidden units per each attention head in each layer')  #
    parser.add_argument('--n_heads', type=str, default='4-2', help='additional entry for the output layer')  #
    parser.add_argument('--residual', type=bool, default=False, help='residual')  #
    parser.add_argument('--act', type=str, default='elu', help='activation funciton')  #

    parser.add_argument('--nb_noising_edges', type=int, default= 20000, help='noisy edges')

    parser.add_argument('--initializer', default='glorot')
    parser.add_argument('--seed',type=int, default=123, help='seed')
    parser.add_argument('--hidden_1', type=int, default=32)
    parser.add_argument('--hidden_2', type=int, default=0)

    parser.add_argument('--topK', type=int, default=5)
    parser.add_argument('--temp_r', type=float, default=1e-3)
    parser.add_argument('--temp_N', type=int, default=50)
    parser.add_argument('--dropout2', default=0.0)

    args, _ = parser.parse_known_args()
    return args

args = get_params()
params = vars(args)


devices = ['0']
if args.machine=='local':
    devices = ['0','1','-1']
elif  args.machine=='dgx':
    devices = ['1','2','5','7']
elif  args.machine=='cpu':
    devices = ['-1']

real_device = args.device%len(devices)
os.environ["CUDA_VISIBLE_DEVICES"] = devices[real_device]
import tensorflow as tf

if tf.__version__.startswith('1.'):
    config = tf.ConfigProto(
        intra_op_parallelism_threads=8,
        inter_op_parallelism_threads=8)

    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

seed = args.seed
random.seed(args.seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

eps = 1e-7
dtype = tf.float32
