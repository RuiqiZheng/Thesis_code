# show the necessity of the portion of under/over sampling
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch

import smote_variants as sv

# %%
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs_gen', type=int, default=10,
                    help='Number of epochs to train for gen.')
parser.add_argument('--ratio_generated', type=float, default=1,
                    help='ratio of generated nodes.')
parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed', 'dblp', 'wiki'], default='cora')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
dataset = args.dataset

# %%

dataset = 'citeseer'
# %%
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_path, dataset_str):
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
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}data/ind.{}.{}".format(dataset_path, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}data/ind.{}.test.index".format(dataset_path, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

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
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
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

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


# adj, features, labels, idx_train, idx_test, minority, majority, majority_test, minority_test = load_data(path=path,
#                                                                                                          dataset=dataset)
# %%
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(
    '/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/dataset/origin_dataset/', 'citeseer')

# if args.cuda:
#     # model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_test = idx_test.cuda()
#     minority = minority.cuda()
#     majority = majority.cuda()

# oversampler= sv.KernelADASYN(proportion=1.5)
# X_samp, y_samp= oversampler.sample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())
#
# print(len(y_samp))
# %%
import scipy

features_numpy = scipy.sparse.lil_matrix.toarray(features)
label = np.zeros((3327, 1), dtype=int)
valid_list = []
empty_label = []
label[0] = 1
for i in range(3327):
    temp_label = []
    # if (labels[i][0] == 0 and labels[i][1] == 0 and labels[i][2] == 0 and labels[i][3] == 0 and labels[i][4] == 0 and
    #         labels[i][5] == 0):
    #     empty_label.append([6])
    for j in range(6):
        if labels[i][j] == 1:
            label[j] = j
            empty_label.append([j])
            print(label[j])
            valid_list.append(i)

valid_features = features_numpy[valid_list]
valid_labels = np.array(empty_label)
# features_numpy_label = np.append(features_numpy, label, axis=1)

# %%
file_path = str(dataset) + "_0.08_0.5.txt"
fo = open(file_path, "a")


# %%

def calculate_0_1(labels):
    labels_1 = labels[np.where(labels == 1)[0]]
    labels_0 = labels[np.where(labels == 0)[0]]
    print('total', len(labels))
    print('0: ', len(labels_0))
    print('1: ', len(labels_1))


# #%%
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=42, sampling_strategy=0.14)
# calculate_0_1(labels[idx_train])
# X_res, y_res = rus.fit_resample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())
# calculate_0_1(y_res)
#
# oversampler = sv.SMOTE(proportion=0.5)
#
# X_samp, y_samp = oversampler.sample(X_res, y_res)
#
# calculate_0_1(y_samp)


# %%

from imblearn.under_sampling import RandomUnderSampler

under_sample_ratios = [round(i * 0.03 + 0.09, 2) for i in range(30)]
over_sample_ratios = [round(i * 0.05 + 0.09, 2) for i in range(30)]
under_sample_ratios = [1]
over_sample_ratios = [1]
# under_sample_ratios = [round(i * 0.002 + 0.9, 3) for i in range(50)]
# over_sample_ratios = [round(i * 0.02 + 0.9, 3) for i in range(50)]
# under_sample_ratios = [0.3]
# over_sample_ratios = [5]
# %%
for under_sample_ratio in under_sample_ratios:
    for over_sample_ratio in over_sample_ratios:
        print(under_sample_ratio, ' ', over_sample_ratio)
        # rus = RandomUnderSampler(random_state=42, sampling_strategy=under_sample_ratio)
        # # calculate_0_1(features_numpy[train_mask])
        # X_res, y_res = rus.fit_resample(features_numpy[train_mask], labels[train_mask])
        # oversampler = sv.SMOTE(proportion=over_sample_ratio)
        # X_samp, y_samp = oversampler.sample(X_res, y_res)
        # calculate_0_1(y_samp)

        # for portion in over_sample_ratios:
        #     fo.write(str(portion) + '\n')
        #     oversampler = sv.KernelADASYN(proportion=portion)
        #     print("portion :", portion)
        #     X_samp, y_samp = oversampler.sample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())

        from sklearn import linear_model

        idx_train = [list(range(0, len(valid_labels) - len(valid_labels) // 8))]
        idx_test = [list(range(len(valid_labels) - len(valid_labels) // 8, len(valid_labels)))]
        logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                                 fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                                 multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                                 solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
        logreg.fit(valid_features[idx_train], valid_labels[idx_train])
        # logreg.fit(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())

        # %%
        # 5. 预测
        import sklearn.metrics

        prepro = logreg.predict(valid_features[idx_test])
        acc = logreg.score(valid_features[idx_test], valid_labels[idx_test])
        print(sklearn.metrics.classification_report(valid_labels[idx_test], valid_features[idx_test],
                                                    labels=[0, 1, 2, 3, 4, 5]))
        # a1 = sklearn.metrics.accuracy_score(labels[idx_test].cpu().numpy(), prepro)
        # a2 = sklearn.metrics.recall_score(labels[idx_test].cpu().numpy(), prepro, pos_label=0)
        # a3 = sklearn.metrics.recall_score(labels[idx_test].cpu().numpy(), prepro)
        # a4 = sklearn.metrics.precision_score(labels[idx_test].cpu().numpy(), prepro, pos_label=0)
        # a5 = sklearn.metrics.precision_score(labels[idx_test].cpu().numpy(), prepro)
        # a6 = sklearn.metrics.f1_score(labels[idx_test].cpu().numpy(), prepro, pos_label=0)
        # a7 = sklearn.metrics.f1_score(labels[idx_test].cpu().numpy(), prepro)
        # a8 = sklearn.metrics.roc_auc_score(labels[idx_test].cpu().numpy(), prepro)
        # # row = {'Accuracy': round(a1, 4), 'Recall0': round(a2, 4), 'Recall1': round(a3, 4), 'Precision0': round(a4, 4),
        # #        'Precision1': round(a5, 4), 'F1-score0': round(a6, 4), 'F1-score1': round(a7, 4), 'AUC': round(a8, 4)}
        # row = [round(a1, 4), round(a2, 4), round(a3, 4), round(a4, 4), round(a5, 4), round(a6, 4), round(a7, 4),
        #        round(a8, 4)]
        # fo.write(str(under_sample_ratio) + " " + str(over_sample_ratio) + " ")
        # fo.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n')
        # fo.write(str(row) + '\n')
        # sys.stdout.flush()
        # print(str(under_sample_ratio) + " " + str(over_sample_ratio))
        # print(row)
        # print("dddd" , prepro)
