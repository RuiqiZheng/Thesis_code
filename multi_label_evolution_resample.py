import copy

import numpy as np
import pygad
import random
from sklearn import linear_model
from sklearn.metrics import classification_report

from comparison.pygcn.pygcn.train import comparison_gcn_train
from evolution_resample import check_percentage, comparison_gcn, draw_violin_plot
from gcn_utils import load_origin_data

global_classification_method = 'GCN'


def change_label_format(labels):
    temp_labels = []
    for i in labels:
        for j in range(len(i)):
            if i[j] == 1:
                temp_labels.append(j)
                continue
        ## citeseer has none label nodes
        sum_temp = np.array(i).sum()
        if np.array(i).sum() != 1:
            temp_labels.append(3)
    temp_labels = np.array(temp_labels)
    return temp_labels


adj, features, labels = load_origin_data('citeseer')
minority_percentage = 0.2

# %%
features = features.toarray()
features_index = np.c_[np.array([i for i in range(0, len(features))]), features]
adj = adj.toarray()
# %%

# %%
labels = change_label_format(labels)
check_percentage(labels)
label_5_delete_index = []
for i in range(len(labels)):
    if labels[i] == 5 and i % 2 == 1:
        label_5_delete_index.append(i)
# %%
label_remain_index = np.delete([i for i in range(len(labels))], label_5_delete_index)
labels = labels[label_remain_index]
features = features[label_remain_index]

adj = np.take(np.take(adj, label_remain_index, axis=0), label_remain_index, axis=1)

check_percentage(labels)
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
def train_logistic_regression_prediction_multi_label(X_samp, y_samp):
    logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                             fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                             multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_samp, y_samp)
    prepro = logreg.predict(features[idx_test])

    return classification_report(labels[idx_test], prepro, output_dict=True)


def fitness_function(solution, solution_idx):
    features_train = features_index[idx_train]
    labels_train = labels[idx_train]

    features_train_1 = features_train[np.where(labels_train == 1)[0]]
    features_train_2 = features_train[np.where(labels_train == 2)[0]]
    features_train_3 = features_train[np.where(labels_train == 3)[0]]
    features_train_4 = features_train[np.where(labels_train == 4)[0]]

    features_train_5 = features_train[np.where(labels_train == 5)[0]]
    features_train_0 = features_train[np.where(labels_train == 0)[0]]
    # features_train_6 = features_train[np.where(labels_train == 6)[0]]

    labels_majorities = [1] * len(features_train_1) + [2] * len(features_train_2) + [3] * len(features_train_3) + [
        4] * len(features_train_4)
    labels_majorities = np.array(labels_majorities)

    # labels_minorities = [0] * len(features_train_0) + [6] * len(features_train_6)
    labels_minorities = [0] * len(features_train_0) + [5] * len(features_train_5)
    labels_minorities = np.array(labels_minorities)

    features_train_majorities = np.concatenate(
        (features_train_1, features_train_2, features_train_3, features_train_4))
    # features_train_minorities = features_train_0
    features_train_minorities = np.concatenate((features_train_0, features_train_5))
    # features_train_minorities = np.concatenate((features_train_1, features_train_6))

    solution_index = []
    for i in range(len(solution)):
        if solution[i] == 0:
            continue
        if solution[i] == 1:
            solution_index.append(i)

    labels_majorities_select = labels_majorities[solution_index]
    features_train_majorities_select = features_train_majorities[solution_index]
    # check_percentage(labels_train)

    y_resample = np.append(labels_majorities_select, labels_minorities)

    X_resample = np.concatenate((features_train_majorities_select, features_train_minorities))
    train_index = X_resample[:, 0:1].reshape(1, -1).astype(int)[0]
    X_resample = X_resample[:, 1:]
    # a = np.concatenate((np.ones(10), np.zeros(10)))
    if global_classification_method == 'LR':
        print(train_logistic_regression_prediction_multi_label(X_resample, y_resample))
        return None
    if solution_idx == -1:
        temp_return = calculate_fitness(X_resample, y_resample, global_classification_method, train_index, True)
    else:
        temp_return = calculate_fitness(X_resample, y_resample, global_classification_method, train_index, False)

    return temp_return


def calculate_fitness(X_resample, y_resample, method, idx_train_m, report_valid=False):
    if method == 'LR':
        report = train_logistic_regression_prediction_multi_label(X_resample, y_resample)
        if report_valid:
            return report
        return ['0']['f1-score']

    else:
        return comparison_gcn(adj, features, labels, idx_test, idx_train_m, 200, report_valid)


def callback_gen(ga_instance):
    # fo = open("/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/log/evolution/multi_label/cora_0408_1311.txt", "a+")
    # fo.write("Generation : ")
    # fo.write("\n")
    # fo.write(str(ga_instance.generations_completed))
    # fo.write("\n")
    # fo.write("Fitness of the best solution :")
    # fo.write("\n")
    # fo.write(str(ga_instance.best_solution()[1]))
    # fo.write("\n")
    # fo.close()
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    # print(ga_instance.best_solution()[0])
    if global_classification_method == 'LR':
        print(fitness_function(ga_instance.best_solution()[0], -1))
    if global_classification_method == 'GCN':
        temp_reports = fitness_function(ga_instance.best_solution()[0], -1)
        print('index of the best solution :', ga_instance.best_solution()[2])


def multi_label_genetic_algorithm(features_train, labels_train,
                                  initial_population_file_name='dataset/pubmed_evolution_initial_population.txt'):
    X_1 = features_train[np.where(labels_train == 1)[0]]
    X_2 = features_train[np.where(labels_train == 2)[0]]
    X_3 = features_train[np.where(labels_train == 3)[0]]
    X_4 = features_train[np.where(labels_train == 4)[0]]
    # X_5 = features_train[np.where(labels_train == 5)[0]]

    num_generations = 500
    num_parents_mating = 20
    sol_per_pop = 50
    parent_selection_type = "rank"  # steady-state selection
    keep_parents = 10
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 0.1

    gene_space = [[[0, 1]] * (len(X_1) + len(X_2) + len(X_3) + len(X_4))][0]
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


def generate_initial_population(features_train, labels_train, population_size=50,
                                file_name='dataset/cora_multi_label_initial_population_50*400.txt'):
    population = []
    # solution = [0] * (len(X_majority) - int(len(X_minority) * 0.9)) + [1] * int(len(X_minority) * 0.9)
    features_train_minority = features_train[np.where(labels_train == 5)[0]]
    minority_count = features_train_minority.shape[0]
    features_train_1 = features_train[np.where(labels_train == 1)[0]]
    features_train_2 = features_train[np.where(labels_train == 2)[0]]
    features_train_3 = features_train[np.where(labels_train == 3)[0]]
    features_train_4 = features_train[np.where(labels_train == 4)[0]]
    # features_train_5 = features_train[np.where(labels_train == 5)[0]]
    for i in range(population_size):
        # temp_solution = copy.deepcopy(solution)
        solution_1 = [0] * (len(features_train_1) - minority_count) + [1] * minority_count
        solution_2 = [0] * (len(features_train_2) - minority_count) + [1] * minority_count
        solution_3 = [0] * (len(features_train_3) - minority_count) + [1] * minority_count
        solution_4 = [0] * (len(features_train_4) - minority_count) + [1] * minority_count
        # solution_5 = [0] * (len(features_train_5) - minority_count) + [1] * minority_count

        random.shuffle(solution_1)
        random.shuffle(solution_2)
        random.shuffle(solution_3)
        random.shuffle(solution_4)
        # random.shuffle(solution_5)
        temp_solution = []
        temp_solution = temp_solution + (solution_1)
        temp_solution = temp_solution + (solution_2)
        temp_solution = temp_solution + (solution_3)
        temp_solution = temp_solution + (solution_4)
        # temp_solution = temp_solution + (solution_5)

        population.append(temp_solution)

    population = np.array(population, dtype=int)
    np.savetxt(fname=file_name, fmt='%i', X=population, delimiter=',')


#
# multi_label_genetic_algorithm(features[idx_train], labels[idx_train],
#                               initial_population_file_name='dataset/pubmed_evolution_initial_population_0.2.txt')

def calculate_initial_population(initial_population_path):
    initial_population = np.loadtxt(fname=initial_population_path, dtype=int, delimiter=',')
    if (len(initial_population.shape) == 1):
        initial_population = initial_population.tolist()
        initial_population = [initial_population]
    else:
        initial_population = initial_population.tolist()
    print(len(initial_population))
    # initial_population_fitness = []
    for i in range(len(initial_population)):
        temp_reports = fitness_function(initial_population[i], -1)
        # initial_population_fitness.append(temp_reports)

        # print("Evaluate {} th solution".format(i))
        print(temp_reports)
    return None
    return initial_population_fitness


def main():
    # comparison_gcn(adj, features, labels, idx_test, idx_train, 400, True)
    # initial_population_fitness = calculate_initial_population('dataset/cora_multi_label_initial_population_50*400.txt')
    # print(calculate_initial_population(
    #     "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multilabel_recall_04081338_best_solution_numpy.txt"))

    # path = '/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/dataset/cora_multi_label_initial_population.txt'
    # initial_population_fitness = calculate_initial_population(path)
    # initial_population_fitness = np.array(initial_population_fitness)
    # np.savetxt(fname='dataset/cora_multi_label_initial_population_fitness_50*400.txt', X=initial_population_fitness, delimiter=',')
    # draw_violin_plot(initial_population_fitness, None, 'pic/evolution/cora_multi_label_F1_50*400.png')
    # generate_initial_population(features[idx_train], labels[idx_train],
    #                             file_name='dataset/citeseer_multi_label_0_5_initial_population_400.txt')
    # initial_population_fitness = calculate_initial_population('dataset/citeseer_multi_label_initial_population_50*400.txt')
    # np.savetxt(fname='dataset/citeseer_multi_label_initial_population_f1_50*400.txt', X=initial_population_fitness, fmt='%s',
    #            delimiter=',')
    multi_label_genetic_algorithm(features_index[idx_train], labels[idx_train],
                                  'dataset/citeseer_multi_label_0_5_initial_population_400.txt')
    # for _ in range(100):
    #     comparison_gcn(adj, features, labels, idx_test, idx_train, 200, True)

main()
# for _ in range(100):
#     comparison_gcn(adj, features, labels, idx_test, idx_train, 200, True)

# report = train_logistic_regression_prediction_multi_label(features[idx_train], labels[idx_train])