from config import args
import tensorflow as tf
from config import *

def process_grads(grads):
    if args.dtype=='float64':
        return grads
    grads = [tf.cast(grad, tf.float16) for grad in grads]
    grads = [tf.cast(grad, tf.float32) for grad in grads]
    return grads
