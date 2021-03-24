# show the necessity of the portion of under/over sampling
from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, add_edges, accuracy_test
from models import GCN_Attention
from models import Generator
from models import GCN
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
parser.add_argument('--dataset', choices=['cora', 'citeseer', 'pubmed', 'dblp', 'wiki'], default='citeseer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# %%

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = args.dataset
# %%

path = "dataset/" + dataset + "/"

if dataset == 'wiki':
    num = 3
else:
    num = 10

# Specfic Parameters to get the best result
if dataset == 'wiki':
    lr = 0.001
elif dataset == 'dblp':
    lr = 0.0009
else:
    lr = 0.01

weight_decay = -1
if dataset == 'cora':
    weight_decay = 0.0001
elif dataset == 'citeseer':
    weight_decay = 0.0005
elif dataset == 'pubmed':
    weight_decay = 0.00008
elif dataset == 'dblp':
    weight_decay = 0.003
elif dataset == 'wiki':
    weight_decay = 0.0005

adj, features, labels, idx_train, idx_test, minority, majority, majority_test, minority_test = load_data(path=path,
                                                                                                         dataset=dataset)
# %%
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()
    minority = minority.cuda()
    majority = majority.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test_majority = accuracy_test(output[idx_test], labels[idx_test], 0)
    acc_test_minority = accuracy_test(output[idx_test], labels[idx_test], 1)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "acc_test_majority= {:.4f}".format(acc_test_majority.item()),
          "acc_test_minority= {:.4f}".format(acc_test_minority.item()))


# Train model
# t_total = time.time()
# for epoch in range(args.epochs):
#     train(epoch)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Testing
# test()

# oversampler= sv.KernelADASYN(proportion=1.5)
# X_samp, y_samp= oversampler.sample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())
#
# print(len(y_samp))
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


# %%

under_sample_ratios = [round(i * 0.03 + 0.09, 2) for i in range(30)]
over_sample_ratios = [round(i * 0.05 + 0.09, 2) for i in range(30)]
# under_sample_ratios = [round(i * 0.002 + 0.9, 3) for i in range(50)]
# over_sample_ratios = [round(i * 0.02 + 0.9, 3) for i in range(50)]
under_sample_ratios = [1]
over_sample_ratios = [1]
# %%
for under_sample_ratio in under_sample_ratios:
    for over_sample_ratio in over_sample_ratios:
        print(under_sample_ratio, ' ', over_sample_ratio)
        rus = RandomUnderSampler(random_state=42, sampling_strategy=under_sample_ratio)
        calculate_0_1(labels[idx_train].cpu().numpy())
        X_res, y_res = rus.fit_resample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())
        oversampler = sv.SMOTE(proportion=over_sample_ratio)
        X_samp, y_samp = oversampler.sample(X_res, y_res)
        calculate_0_1(y_samp)

        # for portion in over_sample_ratios:
        #     fo.write(str(portion) + '\n')
        #     oversampler = sv.KernelADASYN(proportion=portion)
        #     print("portion :", portion)
        #     X_samp, y_samp = oversampler.sample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())

        from sklearn import linear_model

        logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                                 fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                                 multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                                 solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
        logreg.fit(X_samp, y_samp)
        # logreg.fit(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())

        # 5. 预测
        import sklearn.metrics

        prepro = logreg.predict(features[idx_test].cpu().numpy())
        # acc = logreg1.score(X_test1,Y_test1)
        a1 = sklearn.metrics.accuracy_score(labels[idx_test].cpu().numpy(), prepro)
        a2 = sklearn.metrics.recall_score(labels[idx_test].cpu().numpy(), prepro, pos_label=0)
        a3 = sklearn.metrics.recall_score(labels[idx_test].cpu().numpy(), prepro)
        a4 = sklearn.metrics.precision_score(labels[idx_test].cpu().numpy(), prepro, pos_label=0)
        a5 = sklearn.metrics.precision_score(labels[idx_test].cpu().numpy(), prepro)
        a6 = sklearn.metrics.f1_score(labels[idx_test].cpu().numpy(), prepro, pos_label=0)
        a7 = sklearn.metrics.f1_score(labels[idx_test].cpu().numpy(), prepro)
        a8 = sklearn.metrics.roc_auc_score(labels[idx_test].cpu().numpy(), prepro)
        # row = {'Accuracy': round(a1, 4), 'Recall0': round(a2, 4), 'Recall1': round(a3, 4), 'Precision0': round(a4, 4),
        #        'Precision1': round(a5, 4), 'F1-score0': round(a6, 4), 'F1-score1': round(a7, 4), 'AUC': round(a8, 4)}
        row = [round(a1, 4), round(a2, 4), round(a3, 4), round(a4, 4), round(a5, 4), round(a6, 4), round(a7, 4),
               round(a8, 4)]
        fo.write(str(under_sample_ratio) + " " + str(over_sample_ratio) + " ")
        fo.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '\n')
        fo.write(str(row) + '\n')
        sys.stdout.flush()
        print(str(under_sample_ratio) + " " + str(over_sample_ratio))
        print(row)

fo.close()
# %%


# %%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Make data
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
#
# # Plot the surface
# ax.plot_surface(x, y, z)
#
# plt.show()
#
# plt.show()
