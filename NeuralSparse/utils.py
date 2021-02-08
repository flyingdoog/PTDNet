import  numpy as np
import  pickle as pkl
from config import *
import  networkx as nx
import  scipy.sparse as sp
from    scipy.sparse.linalg.eigen.arpack import eigsh
import  sys
import tensorflow as tf
from config import dtype,eps

def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_ppi_data():

    with open('/home/luods/Desktop/SparseGCN/data/ppi.pkl','rb') as fin:
        syn = pkl.load(fin)
    features = syn['features']
    labels = syn['labels']
    adj = syn['adj']
    labels = np.array(labels)
    return adj, features, labels, syn['train_ids'], syn['val_ids'], syn['test_ids']



def load_data(dataset_str,task_type = "semi"):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    if dataset_str == 'ppi':
        return load_ppi_data()

    if dataset_str == 'syn':
        return load_syn_data()

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    return adj, features, labels, idx_train, idx_val, idx_test


def load_syn_data():

    with open('../data/syn.pkl','rb') as fin:
        syn = pkl.load(fin)
    features = syn['features']
    labels = syn['labels']
    adj = syn['adj']

    sizes = features.shape[0]
    nodes = np.array(list(range(sizes)))
    np.random.shuffle(nodes)
    train = int(sizes*0.6)
    val = int(sizes*0.2)
    idx_train = nodes[:train]
    idx_val = nodes[train:train+val]
    idx_test = nodes[train+val:]
    labels = np.array(labels)
    b = np.zeros((labels.size, labels.max() + 1))
    b[np.arange(labels.size), labels] = 1
    return adj, features, b, idx_train, idx_val, idx_test

def load_ppi_data():

    with open('../data/ppi.pkl','rb') as fin:
        syn = pkl.load(fin)
    features = syn['features']
    labels = syn['labels']
    adj = syn['adj']
    labels = np.array(labels)
    return adj, features, labels, syn['train_ids'], syn['val_ids'], syn['test_ids']

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features).astype(np.float32) # D^-1:[2708, 2708]@X:[2708, 2708]
    try:
        return features.todense() # [coordinates, data, shape], []
    except:
        return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)





def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)



def get_adj_list(adj,K):
    N = adj.shape[0]+1 # padding with the last element
    adj_list = np.zeros((N,K),dtype=np.int32)
    adj_list -= 1

    for r in range(N-1): #0-2708
        begin = adj.indptr[r]
        end = adj.indptr[r+1]
        if end-begin >= K:
            # print('maxium nbs',(end-begin))
            end = begin+K
        adj_list[r][:end-begin]=adj.indices[begin:end]
    return adj_list

def get_signed_adj_mask(adj_list, all_labels):
    signed_adj_mask = np.zeros(adj_list.shape,dtype=np.int32)
    N,K = adj_list.shape
    sin_labels = np.argmax(all_labels,axis=-1)

    for i in range(N-1):
        for j in range(K):
            label=-1.0
            if sin_labels[i]==sin_labels[adj_list[i][j]]:
                label =  1.0
            signed_adj_mask[i][j]=label
    return signed_adj_mask

def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random.uniform(
        tf.shape(w),
        minval=eps,
        maxval=1.0,dtype=dtype)
    z = tf.math.log(-tf.math.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t):
    khot_list = []
    onehot_approx = tf.zeros_like(w, dtype=dtype)
    sum_onehot_approx = onehot_approx
    for i in range(k):
        khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        w += tf.math.log(khot_mask)
        onehot_approx = tf.nn.softmax(w / t, axis=-1)
        sum_onehot_approx += onehot_approx
    return sum_onehot_approx



def process_grads(grads):
    if args.dtype=='float64':
        return grads
    grads = [tf.cast(grad, tf.float16) for grad in grads]
    grads = [tf.cast(grad, tf.float32) for grad in grads]
    return grads