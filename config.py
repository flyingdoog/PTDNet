import argparse
import numpy as np
import os
import random


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset", type=str, default='cora', help="Dataset string")
    parser.add_argument('--id', type=str, default='default_id', help='id to store in database')  #
    parser.add_argument('--device', type=int, default=0,help='device to use')  #
    parser.add_argument('--setting', type=str, default="description of hyper-parameters.")  #
    parser.add_argument('--task_type', type=str, default='semi')
    parser.add_argument('--early_stop', type=int, default= 100, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #
    parser.add_argument('--seed',type=int, default=1234, help='seed')
    parser.add_argument('--trails',type=int, default=5, help='trails')



    # shared parameters
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay',type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--hiddens', type=str, default='256')
    parser.add_argument("--lr", type=float, default=0.001,help='initial learning rate.')
    parser.add_argument('--act', type=str, default='leaky_relu', help='activation funciton')  #
    parser.add_argument('--initializer', default='he')
    parser.add_argument('--L', type=int, default=1)  #
    parser.add_argument('--outL', type=int, default=3)  #


    # for dropedge
    parser.add_argument('--dropedge',type=float, default=0., help='dropedge rate (1 - keep probability).')


    # for PTDNet
    parser.add_argument('--init_temperature', type=float, default=2.0)
    parser.add_argument('--temperature_decay', type=float, default=0.99)
    parser.add_argument('--denoise_hidden_1', type=int, default=16)
    parser.add_argument('--denoise_hidden_2', type=int, default=0)
    #
    parser.add_argument('--gamma', type=float, default=-0.0)
    parser.add_argument('--zeta', type=float, default=1.01)

    parser.add_argument('--lambda1', type=float, default=0.1, help='Weight for L0 loss on laplacian matrix.')
    parser.add_argument('--lambda3', type=float, default=0.01, help='Weight for nuclear loss')
    parser.add_argument("--coff_consis", type=float, default=0.01,help='consistency')
    parser.add_argument('--k_svd', type=int, default=1)

    args, _ = parser.parse_known_args()
    return args

args = get_params()
params = vars(args)
SVD_PI = True
devices = ['0','1','-1']
real_device = args.device%len(devices)

os.environ["CUDA_VISIBLE_DEVICES"] = devices[real_device]
import tensorflow as tf

seed = args.seed
random.seed(args.seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

eps = 1e-7

if args.gamma>0:
    print('error gama')
    exit(1)
