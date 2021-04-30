# show the necessity of the portion of under/over sampling
from __future__ import division
from __future__ import print_function
import pygad
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import torch
# from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model
from sklearn.manifold import TSNE
from torch import optim
import torch.nn.functional as F
from utils import load_data
from gcn_utils import load_origin_data
import random
import copy
from sklearn.metrics import classification_report

dataset = 'pubmed'
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


# adj, features, labels, idx_train, idx_test, minority, majority, majority_test, minority_test = load_data(path=path,
#                                                                                                          dataset=dataset)
#
# temp = copy.copy(idx_test)
# idx_test = copy.copy(idx_train)
# idx_train = copy.copy(temp)
# idx_train = idx_train


def change_label_format(labels):
    temp_labels = []
    for i in labels:
        for j in range(len(i)):
            if i[j] == 1:
                temp_labels.append(j)
                continue
    temp_labels = np.array(temp_labels)
    return temp_labels


def multi_label_under_sampling(features, idx_train, labels):
    under_sample_lists = [0, 2, 3, 4, 5]
    under_sample_idx_train = []
    for under_sample_list in under_sample_lists:
        var = np.where(labels == under_sample_list)[0]
        one_label_train_index = list(set(idx_train) & set(var))
        one_label_train_index.sort()
        # under_sample_random = [random.randint(0, len(one_label_train_index)-1) ] * 36
        one_label_train_index = one_label_train_index[0:72]
        under_sample_idx_train = under_sample_idx_train + one_label_train_index

    under_sample_lists = [1, 6]
    for under_sample_list in under_sample_lists:
        var = np.where(labels == under_sample_list)[0]
        one_label_train_index = list(set(idx_train) & set(var))
        one_label_train_index.sort()
        # under_sample_random = [random.randint(0, len(one_label_train_index)-1) ] * 36
        one_label_train_index = one_label_train_index[0:72]
        under_sample_idx_train = under_sample_idx_train + one_label_train_index

    under_sample_idx_train.sort()
    return under_sample_idx_train


def check_percentage(labels):
    count_list = [0] * (int(labels.max()) + 1)
    for i in labels:
        count_list[i] = count_list[i] + 1

    for i in range(len(count_list)):
        print('label {} has {} nodes takes up {}'.format(i, count_list[i], count_list[i] / len(labels)))


adj, features, labels = load_origin_data('cora')
minority_percentage = 0.2
features = features.toarray()
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

# %%
labels_test = labels[idx_test]
check_percentage(labels_test)
# %%
labels_train = labels[idx_train]

check_percentage(labels_train)
# %%
# remove_label_lists = [1, 6]
# for i in remove_label_lists:
#     label_index = np.where(labels == i)[0]
#     label_remove_index = label_index[int(len(label_index) * minority_percentage):-1]
#     for remove_index in label_remove_index:
#         try:
#             idx_train.remove(remove_index)
#         except ValueError:
#             # print("remove {} failed".format(remove_index))
#             a = 1
#
# labels_test = labels[idx_test]
# labels_train = labels[idx_train]
# check_percentage(labels_test)
# check_percentage(labels_train)
# %%


# %%
from comparison.pygcn.pygcn.models import GCN
from comparison.pygcn.pygcn.train import comparison_gcn_train


def comparison_gcn(adj, features, labels, idx_test, idx_train, epoches, report_valid):
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    labels = torch.LongTensor(labels)

    hidden = 16
    dropout = 0.5
    weight_decay = 5e-4
    lr = 0.01
    # temp_values = []
    reports_re = []
    for _ in range(10):
        model = GCN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    dropout=dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=lr, weight_decay=weight_decay)
        model.cuda()
        features_cuda = features.cuda()
        labels_cuda = labels.cuda()
        adj_cuda = adj.cuda()
        idx_train_cuda = torch.tensor(idx_train).cuda()
        idx_test_cuda = torch.tensor(idx_test).cuda()

        for epoch in range(epoches):
            model.train()
            optimizer.zero_grad()
            output = model(features_cuda, adj_cuda)
            loss_train = F.nll_loss(output[idx_train], labels_cuda[idx_train])
            loss_train.backward()
            optimizer.step()

            # print('Epoch: {:04d}'.format(epoch + 1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'acc_train: {:.4f}'.format(acc_train.item()),
            #       'loss_test: {:.4f}'.format(loss_val.item()),
            #       'acc_test: {:.4f}'.format(acc_val.item()),
            #       'time: {:.4f}s'.format(time.time() - t))
        # comparision
        preds = output[idx_test_cuda].max(1)[1].type_as(labels)
        label_report = labels[idx_test_cuda]
        print("comparison:{}".format(
            classification_report(label_report.detach().cpu().numpy(), preds.detach().cpu().numpy(),
                                  output_dict=True)))
        if report_valid:
            preds = output[idx_test_cuda].max(1)[1].type_as(labels)
            label_report = labels[idx_test_cuda]
        else:
            preds = output[idx_train_cuda].max(1)[1].type_as(labels)
            label_report = labels[idx_train_cuda]

        report = classification_report(label_report.detach().cpu().numpy(), preds.detach().cpu().numpy(),
                                       output_dict=True)
        # print(report)

        reports_re.append(report)
        # temp_value = (report['6']['f1-score'] + report['1']['f1-score']) / 2
        # temp_values.append(temp_value)

    minority_reports = {}
    minority_reports['accuracy'] = []
    minority_reports['1'] = {}
    minority_reports['1']['precision'] = []
    minority_reports['1']['recall'] = []
    minority_reports['1']['f1-score'] = []
    minority_reports['6'] = {}
    minority_reports['6']['precision'] = []
    minority_reports['6']['recall'] = []
    minority_reports['6']['f1-score'] = []
    minority_reports['total'] = {}
    minority_reports['total']['precision'] = []
    minority_reports['total']['recall'] = []
    minority_reports['total']['f1-score'] = []

    for key in minority_reports['1']:
        for temp_report in reports_re:
            minority_reports['1'][key].append(temp_report['1'][key])
        # minority_reports['1'][key] = minority_reports['1'][key] / len(reports_re)

    for key in minority_reports['6']:
        for temp_report in reports_re:
            minority_reports['6'][key].append(temp_report['6'][key])
        # minority_reports['6'][key] = minority_reports['6'][key] / len(reports_re)

    for temp_report in reports_re:
        minority_reports['accuracy'].append(temp_report['accuracy'])
    # minority_reports['accuracy'] = minority_reports['accuracy'] / len(reports_re)

    for key in minority_reports['total']:
        for temp_report in reports_re:
            minority_reports['total'][key].append((temp_report['1'][key] + temp_report['6'][key]) / 2)
    if report_valid:
        print('test:{}'.format(minority_reports))

    else:
        print('train:{}'.format(minority_reports))
    # temp_values = np.array(temp_values)
    # print(temp_values)
    # print(temp_values.sum() / len(temp_values))
    temp_np_array = np.array(minority_reports['total']['f1-score'])

    return temp_np_array.sum() / len(temp_np_array)


# comparison_gcn(adj, features, labels, idx_test, idx_train, 200)

# under_sample_idx_train = multi_label_under_sampling(features, idx_train, labels)

# comparison_gcn(adj, features, labels, idx_test, under_sample_idx_train, 200)

# %%
def calculate_0_1(labels):
    labels_1 = labels[np.where(labels == 1)[0]]
    labels_0 = labels[np.where(labels == 0)[0]]
    print('total', len(labels))
    print('0: ', len(labels_0))
    print('1: ', len(labels_1))


def reduce_dimension(X_res, reduce_method):
    if reduce_method == 'pca':
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(X_res)
        X_res_pca = pca.transform(X_res)
        return X_res_pca
    if reduce_method == 'tsne':
        X_res_tsne = TSNE(n_components=2).fit_transform(X_res)
        return X_res_tsne
    else:
        raise TypeError('method \'reduce_dimension\' does not support reduce dimension method other than pca and tsne')


def plot_train_test(X_res, y_res, X_test, y_test, index, info, file_name=None):
    reduce_method = 'tsne'
    X_res_reduce_dimension = reduce_dimension(X_res, reduce_method)
    X_test_reduce_dimension = reduce_dimension(X_test, reduce_method)
    plt.figure(figsize=(8, 8))
    colors = ['navy', 'darkorange']
    test_colors = ['khaki', 'grey']

    # pca.fit(X_res_test)
    # X_res_test_pca = pca.transform(X_res_test)
    if X_test is not None and y_test is not None:
        for color, i, target_name in zip(test_colors, [0, 1], ['test_majority', 'test_minority']):
            plt.scatter(X_test_reduce_dimension[y_test == i, 0], X_test_reduce_dimension[y_test == i, 1],
                        color=color, lw=0.01, label=target_name)

    for color, i, target_name in zip(colors, [0, 1], ['majority', 'minority']):
        plt.scatter(X_res_reduce_dimension[y_res == i, 0], X_res_reduce_dimension[y_res == i, 1],
                    color=color, lw=0.01, label=target_name)

    plt.title("undersampling on Pubmed and test data")

    for parameter in info:
        plt.plot([], [], ' ', label=str(parameter) + ": " + str(info[parameter]))
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    if file_name is None:
        plt.savefig('pic/evolution/Pubmed_undersample_{}_{}.png'.format(index, reduce_method), dpi=300)
    else:
        plt.savefig(file_name, dpi=300)
    plt.show()


def train_logistic_regression(X_samp, y_samp):
    temp, _ = train_logistic_regression_prediction(X_samp, y_samp)
    return temp


def train_logistic_regression_prediction_multilabel(X_samp, y_samp):
    logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                             fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                             multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_samp, y_samp)
    prepro = logreg.predict(features[idx_test])
    report = classification_report(labels[idx_test], prepro, output_dict=True)
    print(classification_report(labels[idx_test], prepro))
    print(1)
    return report


def train_logistic_regression_prediction(X_samp, y_samp):
    logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                             fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                             multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_samp, y_samp)

    prepro = logreg.predict(features[idx_test])
    # acc = logreg1.score(X_test1,Y_test1)
    a1 = sklearn.metrics.accuracy_score(labels[idx_test], prepro)
    a2 = sklearn.metrics.recall_score(labels[idx_test], prepro, pos_label=0)
    a3 = sklearn.metrics.recall_score(labels[idx_test], prepro)
    a4 = sklearn.metrics.precision_score(labels[idx_test], prepro, pos_label=0)
    a5 = sklearn.metrics.precision_score(labels[idx_test], prepro)
    a6 = sklearn.metrics.f1_score(labels[idx_test], prepro, pos_label=0)
    a7 = sklearn.metrics.f1_score(labels[idx_test], prepro)
    a8 = sklearn.metrics.roc_auc_score(labels[idx_test], prepro)
    row = {'Accuracy': round(a1, 4), 'Recall0': round(a2, 4), 'Recall1': round(a3, 4), 'Precision0': round(a4, 4),
           'Precision1': round(a5, 4), 'F1-score0': round(a6, 4), 'F1-score1': round(a7, 4), 'AUC': round(a8, 4)}
    # row = [round(a1, 4), round(a2, 4), round(a3, 4), round(a4, 4), round(a5, 4), round(a6, 4), round(a7, 4),
    #        round(a8, 4)]
    return row, prepro


def callback_gen(ga_instance):
    fo = open("log/evolution/evolution_pubmed_0324_2044.txt", "a+")
    fo.write("Generation : ")
    fo.write("\n")
    fo.write(str(ga_instance.generations_completed))
    fo.write("\n")
    fo.write("Fitness of the best solution :")
    fo.write("\n")
    fo.write(str(ga_instance.best_solution()[1]))
    fo.write("\n")
    fo.close()
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


def draw_violin_plot(data1, data2, file_name):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axs[0].violinplot(data1,
                      showmeans=True,
                      showmedians=True)
    axs[0].set_title('initial_population_50')
    plt.plot([], [], ' ', label="max fitness: {}".format(np.array(data1).max()))
    if data2 is not None:
        axs[1].violinplot(data2,
                          showmeans=True,
                          showmedians=True)
        axs[1].set_title('initial_population_1500')
    if file_name is None:
        file_name = 'pic/evolution/Pubmed_undersample_1500_epoch.png'
    plt.savefig(file_name, dpi=200)
    plt.show()


def calculate_initial_population(initial_population_path):
    initial_population = np.loadtxt(fname=initial_population_path, dtype=int, delimiter=',')
    initial_population = initial_population.tolist()
    initial_population_fitness = []
    for solution in initial_population:
        initial_population_fitness.append(fitness_function(solution, None))
    return initial_population_fitness


def fitness_function(solution, solution_idx):
    global features, labels
    _features = features[idx_train]
    _labels = labels[idx_train]
    X_majority = _features[np.where(_labels == 0)[0]]
    X_minority = _features[np.where(_labels == 1)[0]]
    y_majority = _labels[np.where(_labels == 0)[0]]
    y_minority = _labels[np.where(_labels == 1)[0]]
    solution_index = []

    for i in range(len(solution)):
        if solution[i] == 0:
            continue
        if solution[i] == 1:
            solution_index.append(i)

    X_majority = X_majority[solution_index]
    y_majority = y_majority[solution_index]
    y_resample = np.append(y_majority, y_minority)
    X_resample = np.concatenate((X_majority, X_minority))
    # a = np.concatenate((np.ones(10), np.zeros(10)))

    fitness = train_logistic_regression(X_resample, y_resample)
    return float(fitness['F1-score1'])


# %%
def genetic_algorithm(X, y, initial_population_file_name='dataset/pubmed_evolution_initial_population.txt'):
    X_majority = X[np.where(y == 0)[0]]
    X_minority = X[np.where(y == 1)[0]]
    num_generations = 5000
    num_parents_mating = 20
    sol_per_pop = 50
    parent_selection_type = "rank"  # steady-state selection
    keep_parents = 10
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 0.1

    gene_space = [[[0, 1]] * len(X_majority)][0]
    num_genes = len(gene_space)
    initial_population = np.loadtxt(fname=initial_population_file_name, dtype=int, delimiter=',')

    initial_population = initial_population.tolist()
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           gene_type=int,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           on_generation=callback_gen,
                           mutation_percent_genes=mutation_percent_genes,
                           initial_population=initial_population)

    ga_instance.run()
    ga_instance.plot_result()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


def generate_initial_population(X, y, population_size=50, file_name='dataset/pubmed_evolution_initial_population.txt'):
    X_majority = X[np.where(y == 0)[0]]
    X_minority = X[np.where(y == 1)[0]]
    population = []
    solution = [0] * (len(X_majority) - int(len(X_minority) * 0.9)) + [1] * int(len(X_minority) * 0.9)

    for i in range(population_size):
        temp_solution = copy.deepcopy(solution)
        random.shuffle(temp_solution)
        population.append(temp_solution)
        temp_solution = []
    population = np.array(population, dtype=int)
    np.savetxt(fname=file_name, fmt='%i', X=population, delimiter=',')


def random_under_sample():
    for i in range(2):
        # rus = RandomUnderSampler(random_state=i, sampling_strategy=0.9)
        # X_res, y_res = rus.fit_resample(features[idx_train], labels[idx_train])
        X_res_test, y_res_test = features[idx_test], labels[idx_test]
        row, prepro = train_logistic_regression_prediction(features[idx_train], labels[idx_train])
        # plot_train_test(X_res, y_res, X_res_test, prepro, i, row,
        #                 'pic/evolution/{}_undersample_test_train_seed{}.png'.format(dataset, i))
        # plot_train_test(X_res, y_res, X_test=None, y_test=None, index=i, info=row)
        print(train_logistic_regression(features[idx_train], labels[idx_train]))


def main():
    under_sample_idx_train = multi_label_under_sampling(features, idx_train, labels)
    # train_logistic_regression_prediction_multilabel(features[under_sample_idx_train], labels[under_sample_idx_train])
    report = train_logistic_regression_prediction_multilabel(features[idx_train], labels[idx_train])
    # X = features[idx_train]).numpy()
    # y = labels[idx_train]).numpy()
    # 
    # if False:
    #     generate_initial_population(X, y)
    # generate_initial_population(features[idx_train], labels[idx_train], 50 * 110,
    #                             'dataset/pubmed_evolution_initial_population_0.2_50*110.txt')
    # initial_population_fitness = calculate_initial_population(
    #     'dataset/pubmed_evolution_initial_population_0.2.txt')

    # print(initial_population_fitness)
    # initial_population_1500_fitness = calculate_initial_population('dataset/evolution_initial_population_1500.txt')
    # draw_violin_plot(initial_population_fitness, None, 'pic/evolution/Pubmed_undersample_0.2_50*110.png')
    # initial_population_fitness = np.array(initial_population_fitness)
    # max = initial_population_fitness.max()
    # min = initial_population_fitness.min()
    # print("max: {}, min: {}".format(max, min))
    # genetic_algorithm(features[idx_train], labels[idx_train],
    #                   initial_population_file_name='dataset/pubmed_evolution_initial_population_0.2.txt')
    # X_res, y_res = rus.fit_resample(features[idx_train]).numpy(), labels[idx_train]).numpy())
    # X_res_test, y_res_test = features[idx_test][[0, 1, 2]]).numpy(), labels[idx_test][[0, 1, 2]]).numpy()
    # calculate_0_1(y_res)
    # random_under_sample()
