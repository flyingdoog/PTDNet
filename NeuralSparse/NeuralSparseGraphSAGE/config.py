import  argparse
import  os
import numpy as np
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--multi_label',type= bool, default=False)

args.add_argument('--model', default='graphsage_mean',help='model names. See README for possible values.')
args.add_argument('--task_type', default='semi')
args.add_argument('--seed', type= int, default=123)

args.add_argument('--learning_rate',type=float, default=0.01)
args.add_argument('--epochs', type=int,default=3000)
args.add_argument('--samples_1',type=int, default=32,help='number of samples in layer 1')
args.add_argument('--samples_2',type = int, default=16,help='number of samples in layer 2')
args.add_argument('--dropout', type = float,default=0.1)
args.add_argument('--dim_1', type = int,default=128,help='Size of output dim (final is 2x this, if using concat)')
args.add_argument('--dim_2', type = int,default=64,help='Size of output dim (final is 2x this, if using concat)')
args.add_argument('--print_every', type=int, default=5)
args.add_argument('--use_gpu', type=str, default="0")  #
args.add_argument('--device', type=int, default=0)  #
args.add_argument('--dtype', type=str, default='float32')  #
args.add_argument('--setting', type=str, default="test only")  #
args.add_argument('--id', type=str, default="hello test only")  #
args.add_argument('--val_batch_size', type=int, default=128)

args.add_argument('--temp_r',type=float, default=1e-3)
args.add_argument('--temp_N',type=int, default=50)


args.add_argument('--batch_size', default=256, help='minibatch size.')
args.add_argument('--nb_noising_edges', type=int, default=20000)


args.add_argument('--weight_decay', default=0)
args.add_argument('--early_stopping', default=100)
args.add_argument('--max_degree', default=32)

args.add_argument('--dropout2', default=0.0)
args.add_argument('--hidden_1', type=int, default=32)
args.add_argument('--hidden_2', type=int, default=0)
args.add_argument('--act', default='relu')
args.add_argument('--initializer', default='glorot')

args.add_argument('--topK', type=int, default=5)
args.add_argument('--weighted', type=bool, default=True)

args = args.parse_args()
print(args)
params = vars(args)
need_check_edge = True
val_whole_batch = True

devices = ['0']

real_device = args.device%len(devices)
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(args.seed)

eps = 1e-8

dtype = tf.float32