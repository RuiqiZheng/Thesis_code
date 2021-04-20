import numpy as np
from gcn_utils import load_origin_data
from evolution_resample import change_label_format
from sklearn import linear_model
from sklearn.metrics import classification_report

adj, features, labels = load_origin_data('cora')
minority_percentage = 0.2
features = features.toarray()
features_index = np.c_[np.array([i for i in range(0, len(features))]), features]
adj = adj.toarray()
labels = change_label_format(labels)

print(1)
# %%
# 划分测试集训练集
test_split_ratio = 0.7
diffrent_labels_index = []
for i in range(labels.max() + 1):
    diffrent_labels_index.append(np.where(labels == i)[0])
idx_test = []
idx_train = []
for one_label_index in diffrent_labels_index:
    idx_test = idx_test + \
               one_label_index[int(len(one_label_index) * (1 - test_split_ratio)):-1].reshape(1, -1).tolist()[0]
    idx_train = idx_train + \
                one_label_index[0: int(len(one_label_index) * (1 - test_split_ratio))].reshape(1, -1).tolist()[0]

# idx_test = [i for i in range(1000, len(labels))]
# idx_train = [i for i in range(1000)]
idx_test.sort()
idx_train.sort()


def train_logistic_regression_prediction_multi_label(X_samp, y_samp):
    logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                             fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                             multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_samp, y_samp)
    prepro = logreg.predict(features[idx_test])

    return classification_report(labels[idx_test], prepro)


features_train = features[idx_train]
labels_train = labels[idx_train]
features_train_0 = features_train[np.where(labels_train == 0)[0]]
features_train_2 = features_train[np.where(labels_train == 2)[0]]
features_train_1 = features_train[np.where(labels_train == 1)[0]]

features_train_3 = features_train[np.where(labels_train == 3)[0]]
features_train_4 = features_train[np.where(labels_train == 4)[0]]

features_train_1_2 = np.concatenate((features_train_1, features_train_2))
label_1_2 = [1] * len(features_train_1) + [2] * len(features_train_2)

features_train_5 = features_train[np.where(labels_train == 5)[0]]
features_train_6 = features_train[np.where(labels_train == 6)[0]]
features_train_5_6 = np.concatenate((features_train_5, features_train_6))
label_5_6 = [5] * len(features_train_5) + [6] * len(features_train_6)

import smote_variants as sv

oversampler = sv.SMOTE()
features_train_1_2, label_1_2 = oversampler.sample(features_train_1_2, label_1_2)

features_train_5_6, label_5_6 = oversampler.sample(features_train_5_6, label_5_6)

X_resample = np.concatenate(
    (features_train_0, features_train_1_2, features_train_3, features_train_4, features_train_5_6))

y_resample = np.append([0] * len(features_train_0), label_1_2)
y_resample = np.append(y_resample, [3] * len(features_train_3))
y_resample = np.append(y_resample, [4] * len(features_train_4))
y_resample = np.append(y_resample, label_5_6)


print(train_logistic_regression_prediction_multi_label(X_resample, y_resample))
