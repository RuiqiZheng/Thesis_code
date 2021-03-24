# show the necessity of the portion of under/over sampling
from __future__ import division
from __future__ import print_function
import pygad
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model
from sklearn.manifold import TSNE
from utils import load_data
import random
import copy

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

adj, features, labels, idx_train, idx_test, minority, majority, majority_test, minority_test = load_data(path=path,
                                                                                                         dataset=dataset)


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


def plot(X_res, y_res, index, info):
    reduce_method = 'tsne'
    X_res_reduce_dimension = reduce_dimension(X_res, reduce_method)
    plt.figure(figsize=(8, 8))
    colors = ['navy', 'darkorange']
    test_colors = ['black', 'grey']

    # pca.fit(X_res_test)
    # X_res_test_pca = pca.transform(X_res_test)

    for color, i, target_name in zip(colors, [0, 1], ['majority', 'minority']):
        plt.scatter(X_res_reduce_dimension[y_res == i, 0], X_res_reduce_dimension[y_res == i, 1],
                    color=color, lw=0.01, label=target_name)

    # for color, i, target_name in zip(test_colors, [0, 1], ['test_majority', 'test_minority']):
    #         plt.scatter(X_res_test_pca[y_res_test == i, 0], X_res_test_pca[y_res_test == i, 1],
    #                     color=color, lw=2, label=target_name)
    plt.title("undersampling on Pubmed")

    for parameter in info:
        plt.plot([], [], ' ', label=str(parameter) + ": " + str(info[parameter]))
    plt.legend(loc="best", shadow=False, scatterpoints=1)

    plt.savefig('pic/evolution/Pubmed_undersample_{}_{}.png'.format(index, reduce_method), dpi=300)
    plt.show()


def train_logistic_regression(X_samp, y_samp):
    logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                             fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                             multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_samp, y_samp)

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
    row = {'Accuracy': round(a1, 4), 'Recall0': round(a2, 4), 'Recall1': round(a3, 4), 'Precision0': round(a4, 4),
           'Precision1': round(a5, 4), 'F1-score0': round(a6, 4), 'F1-score1': round(a7, 4), 'AUC': round(a8, 4)}
    # row = [round(a1, 4), round(a2, 4), round(a3, 4), round(a4, 4), round(a5, 4), round(a6, 4), round(a7, 4),
    #        round(a8, 4)]
    return row


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

def calculate_initial_population(initial_population_path):
    initial_population = np.loadtxt(fname=initial_population_path, dtype=int, delimiter=',')

def fitness_function(solution, solution_idx):
    global features, labels
    _features = features[idx_train].cpu().numpy()
    _labels = labels[idx_train].cpu().numpy()
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
    return float(fitness['Recall1'])


# %%
def genetic_algorithm(X, y):
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
    initial_population = np.loadtxt(fname='dataset/evolution_initial_population.txt', dtype=int, delimiter=',')

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
                           callback_generation=callback_gen,
                           mutation_percent_genes=mutation_percent_genes,
                           initial_population=initial_population)

    ga_instance.run()
    ga_instance.plot_result()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


def generate_initial_population(X, y, population_size=50):
    X_majority = X[np.where(y == 0)[0]]
    X_minority = X[np.where(y == 1)[0]]
    population = []
    solution = [0] * (len(X_majority) - int(len(X_minority) // 0.9)) + [1] * int(len(X_minority) * 0.9)

    for i in range(population_size):
        temp_solution = copy.deepcopy(solution)
        random.Random(i).shuffle(temp_solution)
        population.append(temp_solution)
        temp_solution = []
    population = np.array(population, dtype=int)
    np.savetxt(fname='dataset/evolution_initial_population.txt', fmt='%i', X=population, delimiter=',')


for i in range(1):
    # rus = RandomUnderSampler(random_state=i, sampling_strategy=0.9)
    X = features[idx_train].cpu().numpy()
    y = labels[idx_train].cpu().numpy()
    if False:
        generate_initial_population(X, y)
    genetic_algorithm(X, y)
    # X_res, y_res = rus.fit_resample(features[idx_train].cpu().numpy(), labels[idx_train].cpu().numpy())
    # X_res_test, y_res_test = features[idx_test][[0, 1, 2]].cpu().numpy(), labels[idx_test][[0, 1, 2]].cpu().numpy()
    # calculate_0_1(y_res)

    # raw = train_logistic_regression(X_res, y_res)
    # plot(X_res, y_res, i, raw)
