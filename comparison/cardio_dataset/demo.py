import copy

import pygad
import sklearn
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import classification_report
import random
from sklearn.decomposition import PCA

dat = io.loadmat('cardio.mat')
key_list = list(dat.keys())

X = dat['X']
y = dat['y'].astype(np.int32)
y = y.reshape(1, -1)[0]
features = X
labels = y

#
#
# from sklearn.manifold import TSNE
# fig, ax = plt.subplots()
# temp_label1 = []
# temp_label0 = []
# print("begin tsne")
# # X_embedded = TSNE(n_components=2, random_state=1).fit_transform(features)
# X_embedded = PCA(n_components=2, random_state=1).fit_transform(features)
#
# print("finish tsne")
# for data, label in zip(X_embedded, labels):
#     if label > 0:
#         temp_label1.append(data)
#     else:
#         temp_label0.append(data)
# temp_label1 = np.array(temp_label1)
# temp_label0 = np.array(temp_label0)
# ax.scatter(temp_label1[:, 0], temp_label1[:, 1], s=5, color={'red'}, label='train set label 1')
# ax.scatter(temp_label0[:, 0], temp_label0[:, 1], s=5, color={'blue'}, label='train set label 0')
#
# ax.legend()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("train and test.png", dpi=300)
# plt.show()
# plt.clf()

picture_count = 0

idx_train = []
idx_test = []
idx_0 = []
idx_1 = []
count = 0
for data, label in zip(X, y):

    if label == 1:
        idx_0.append(count)
    if label == 0:
        idx_1.append(count)
    count = count + 1
random.seed(1)
random.Random(4).shuffle(idx_0)
random.Random(4).shuffle(idx_1)
idx_train = idx_0[:int(len(idx_0) * 0.3)] + idx_1[:int(len(idx_1) * 0.3)]
idx_test = idx_0[int(len(idx_0) * 0.3):] + idx_1[int(len(idx_1) * 0.3):]

from sklearn.manifold import TSNE

fig, ax = plt.subplots()
temp_label1 = []
temp_label0 = []
print("begin tsne")
# X_embedded = TSNE(n_components=2, random_state=1).fit_transform(features[idx_train])
tsne = TSNE(n_components=2, random_state=1)
pca = PCA(n_components=2, random_state=1)
pca.fit(features[idx_train])
X_embedded = pca.transform(features[idx_train])
print("finish tsne")
for data, label in zip(X_embedded, labels[idx_train]):
    if label > 0:
        temp_label1.append(data)
    else:
        temp_label0.append(data)
temp_label1 = np.array(temp_label1)
temp_label0 = np.array(temp_label0)
# random.Random(130).shuffle(temp_label0)
# temp_label0 = temp_label0[:len(temp_label1)]
ax.scatter(temp_label1[:, 0], temp_label1[:, 1], s=5, color={'tab:red'}, label='train set label 1', alpha=0.6)
ax.scatter(temp_label0[:, 0], temp_label0[:, 1], s=5, color={'tab:blue'}, label='train set label 0', alpha=0.6)

ax.legend()
plt.xlim([-6, 10])
plt.ylim([-6, 6])
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("train.pdf")
plt.show()
plt.clf()

def simple_oversamping():
    fig, ax = plt.subplots()
    temp_label1 = []
    temp_label0 = []
    print("begin tsne")
    # X_embedded = TSNE(n_components=2, random_state=1).fit_transform(features[idx_train])
    tsne = TSNE(n_components=2, random_state=1)
    pca = PCA(n_components=2, random_state=1)
    pca.fit(features[idx_train])
    X_embedded = pca.transform(features[idx_train])
    print("finish tsne")
    for data, label in zip(X_embedded, labels[idx_train]):
        if label > 0:
            temp_label1.append(data)
        else:
            temp_label0.append(data)
    temp_label1 = np.array(temp_label1)
    temp_label0 = np.array(temp_label0)
    temp_label1_over = copy.deepcopy(temp_label1)
    for _ in range(9):
        temp_label1_over = np.concatenate((temp_label1_over, temp_label1))
    ax.scatter(temp_label1_over[:, 0], temp_label1[:, 1], s=100, color={'tab:red'}, label='train set label 1', alpha=0.6)
    ax.scatter(temp_label0[:, 0], temp_label0[:, 1], s=5, color={'tab:blue'}, label='train set label 0', alpha=0.6)

    ax.legend()
    plt.xlim([-6, 10])
    plt.ylim([-6, 6])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("0_oversampling.pdf")
    plt.show()
    plt.clf()


def best_fit():
    best_solution = "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 1 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]"
    best_solution = best_solution[1:-2]
    best_solution = best_solution.split(' ')
    best_solution = best_solution[:-1]
    temp_list = []
    for i in best_solution:
        temp_list.append(int(i))
    temp_list = np.array(temp_list)
    fitness_function(temp_list, -2)


def undersampling():
    fig, ax = plt.subplots()
    temp_label1 = []
    temp_label0 = []
    print("begin tsne")
    # X_embedded = TSNE(n_components=2, random_state=1).fit_transform(features[idx_train])
    tsne = TSNE(n_components=2, random_state=1)
    pca = PCA(n_components=2, random_state=1)
    pca.fit(features[idx_train])
    X_embedded = pca.transform(features[idx_train])
    print("finish tsne")
    for data, label in zip(X_embedded, labels[idx_train]):
        if label > 0:
            temp_label1.append(data)
        else:
            temp_label0.append(data)
    temp_label1 = np.array(temp_label1)
    temp_label0 = np.array(temp_label0)
    temp_label0_outlier = []
    temp_label0_normal = []
    for i in temp_label0:
        #
        if i[0] + i[1] < -3 or i[1] - i[0] < -4 or i[1] > 3 or i[0] < -3:
            temp_label0_outlier.append(i)
        else:
            temp_label0_normal.append(i)
    temp_label0_outlier = np.array(temp_label0_outlier)
    temp_label0_normal = np.array(temp_label0_normal)
    temp_label0_outlier = temp_label0_outlier[:len(temp_label1) // 4 * 3]
    temp_label0_normal = temp_label0_normal[:len(temp_label1) // 4]
    temp_label0 = np.concatenate((temp_label0_outlier, temp_label0_normal))
    # temp_label0 = temp_label0[:len(temp_label1)]
    # random.Random(130).shuffle(temp_label0)
    # temp_label0 = temp_label0[:len(temp_label1)]
    ax.scatter(temp_label1[:, 0], temp_label1[:, 1], s=5, color={'tab:red'}, label='train set label 1', alpha=0.6)
    ax.scatter(temp_label0[:, 0], temp_label0[:, 1], s=5, color={'tab:blue'}, label='train set label 0', alpha=0.6)

    ax.legend()
    plt.xlim([-6, 10])
    plt.ylim([-6, 6])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("random_undersampling.pdf")
    plt.show()
    plt.clf()


def smote_pic():
    fig, ax = plt.subplots()
    import smote_variants as sv
    temp_label1 = []
    temp_label0 = []
    oversampler = sv.SMOTE()
    oversampler = sv.ADASYN()

    features_train_smote, label_smote = oversampler.sample(features[idx_train], labels[idx_train])

    X_embedded = pca.transform(features_train_smote)
    print("finish tsne")
    for data, label in zip(X_embedded, label_smote):
        if label > 0:
            temp_label1.append(data)
        else:
            temp_label0.append(data)
    temp_label1 = np.array(temp_label1)
    temp_label0 = np.array(temp_label0)
    # random.Random(130).shuffle(temp_label0)
    # temp_label0 = temp_label0[:len(temp_label1)]
    ax.scatter(temp_label1[:, 0], temp_label1[:, 1], s=5, color={'tab:red'}, label='train set label 1', alpha=0.6)
    ax.scatter(temp_label0[:, 0], temp_label0[:, 1], s=5, color={'tab:blue'}, label='train set label 0', alpha=0.6)

    ax.legend()
    plt.xlim([-6, 10])
    plt.ylim([-6, 6])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("pdf/ADASYN.pdf")
    plt.show()
    plt.clf()
    return 0

def train_logistic_regression(X_samp, y_samp, solution_idx):
    logreg = linear_model.LogisticRegression(C=100000.0, class_weight=None, dual=False,
                                             fit_intercept=True, intercept_scaling=1, max_iter=1500,
                                             multi_class='auto', n_jobs=None, penalty='l2', random_state=None,
                                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    logreg.fit(X_samp, y_samp)
    if solution_idx == -1 or solution_idx % 20 == 0 or solution_idx == -2:
        from sklearn.manifold import TSNE
        fig, ax = plt.subplots()
        temp_label1 = []
        temp_label0 = []
        print("begin tsne")
        # X_embedded = TSNE(n_components=2, random_state=1).fit_transform(X_samp)
        X_embedded = pca.transform(X_samp)
        print("finish tsne")
        for data, label in zip(X_embedded, y_samp):
            if label > 0:
                temp_label1.append(data)
            else:
                temp_label0.append(data)
        temp_label1 = np.array(temp_label1)
        temp_label0 = np.array(temp_label0)
        ax.scatter(temp_label1[:, 0], temp_label1[:, 1], s=5, color={'red'}, label='train set label 1')
        ax.scatter(temp_label0[:, 0], temp_label0[:, 1], s=5, color={'blue'}, label='train set label 0')

        ax.legend()
        plt.xlim([-6, 10])
        plt.ylim([-6, 6])
        plt.xlabel("X")
        plt.ylabel("Y")
        global picture_count
        picture_count = picture_count + 1
        if solution_idx == -1:
            plt.title("current best solution in generation {}".format(int(picture_count / 50)))
        else:
            plt.title("individual solution in generation 100".format(int(picture_count / 50)))
        plt.savefig("pdf/1_train_genetic_epoch_{}_{}.pdf".format(picture_count, solution_idx))
        plt.show()
        plt.clf()

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
    report = classification_report(labels[idx_test], prepro, output_dict=True)
    # print(1)
    return report['1']['f1-score']


def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    if ga_instance.generations_completed % 1 == 0:
        fitness_function(ga_instance.best_solution()[0], -1)
        print("Begin draw pictures")


def draw_violin_plot(data1, data2):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axs[0].violinplot(data1,
                      showmeans=True,
                      showmedians=True)
    axs[0].set_title('initial_population_50')
    axs[1].violinplot(data2,
                      showmeans=True,
                      showmedians=True)
    axs[1].set_title('initial_population_1500')
    plt.savefig('pic/evolution/Pubmed_undersample_1500_epoch.png', dpi=200)
    plt.show()


def calculate_initial_population(initial_population_path):
    initial_population = np.loadtxt(fname=initial_population_path, dtype=int, delimiter=',')
    initial_population = initial_population.tolist()
    initial_population_fitness = []
    for solution in initial_population:
        initial_population_fitness.append(fitness_function(solution, None))
    return initial_population_fitness


def fitness_function(solution, solution_idx):
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

    fitness = train_logistic_regression(X_resample, y_resample, solution_idx)
    return fitness


# %%
def genetic_algorithm(X, y):
    X_majority = X[np.where(y == 0)[0]]
    X_minority = X[np.where(y == 1)[0]]
    num_generations = 100
    num_parents_mating = 20
    sol_per_pop = 50
    parent_selection_type = "rank"  # steady-state selection
    keep_parents = 10
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 0.1

    gene_space = [[[0, 1]] * len(X_majority)][0]
    num_genes = len(gene_space)
    initial_population = np.loadtxt(fname='cardio_initial.txt', dtype=int, delimiter=',')

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


def generate_initial_population(X, y, population_size=50, file_name='cardio_initial.txt'):
    X_majority = X[np.where(y == 0)[0]]
    X_minority = X[np.where(y == 1)[0]]
    population = []
    solution = [0] * (len(X_majority) - int(len(X_minority) * 0.9)) + [1] * int(len(X_minority) * 0.9)

    for i in range(population_size):
        temp_solution = copy.deepcopy(solution)
        random.Random(i).shuffle(temp_solution)
        population.append(temp_solution)
        temp_solution = []
    population = np.array(population, dtype=int)
    np.savetxt(fname=file_name, fmt='%i', X=population, delimiter=',')


# generate_initial_population(features[idx_train], labels[idx_train])
# genetic_algorithm(features[idx_train], labels[idx_train])
# undersampling()
# best_fit()
# smote_pic()
# simple_oversamping()
smote_pic()