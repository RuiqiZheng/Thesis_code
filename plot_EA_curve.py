import matplotlib.pyplot as plt

path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/citeseer_multi_label_0_5_gcn_04301745.txt"

file1 = open(path, 'r')
Lines = file1.readlines()
count = 0
generations = []
fitness = []
# Strips the newline character
count = 0
for line in Lines:
    # if line.startswith('Generation'):
    #     generations.append(int(line.split('  ')[-1]))
    #
    # if line.startswith(('Fitness')):
    #     fitness.append(float(line.split(' ')[-1]))
    count = count + 1
    print(line)
    if count == 100:
        break

print(fitness)



# %%
# plotting the points
path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multilabel_04081334.txt"

file1 = open(path, 'r')
Lines = file1.readlines()
count = 0
generations = []
fitness = []
# Strips the newline character
for line in Lines:
    if line.startswith('Generation'):
        generations.append(int(line.split('  ')[-1]))

    if line.startswith(('Fitness')):
        fitness.append(float(line.split(' ')[-1]))

print(fitness)

# %%
import numpy as np


def plot_random_fitness(path):
    random_fitness = np.loadtxt(fname=path, delimiter=',')
    return random_fitness


gcn_distribution = [0.7100271002710027, 0.6611764705882353, 0.7013189754046132, 0.6704073616878017,
                    0.713006029285099, 0.6557943335436942, 0.6668051861702128, 0.7128899368512347] * 50

random_fitness = plot_random_fitness(
    "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/dataset/cora_multi_label_initial_population_F1score_fitness_50*400.txt")
random_fitness = random_fitness.reshape(1, -1)[0]
random_fitness_list = []
temp_max = 0.5791738745200625
i = 0
for i in range(int(len(random_fitness) / 50)):
    temp_random_fitness = random_fitness[i * 50:(i + 1) * 50]
    temp_max = max(temp_random_fitness.max(), temp_max)
    random_fitness_list.append(temp_max)

random_fitness_list_index = [i for i in range(len(random_fitness_list))]
# %%
temp_gcn_genetic_fitness = 0
path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multi_label_f1_gcn_04200316.txt"
gcn_genetic_fitness = []
file1 = open(path, 'r')
Lines = file1.readlines()

# Strips the newline character
for line in Lines:
    if line.startswith('Fitness of the best solution :'):
        temp_fitness = float(line.split('\n')[0].split(' ')[-1])
        temp_gcn_genetic_fitness = max(temp_gcn_genetic_fitness, temp_fitness)
        gcn_genetic_fitness.append(temp_gcn_genetic_fitness)

gcn_genetic_index = [i for i in range(len(gcn_genetic_fitness))]

fig, ax = plt.subplots(figsize=(9, 9))
if len(generations) != len(fitness):
    print("length of generation: {}, length of fitness: {}".format(len(generations), len(fitness)))
    temp_min = min(len(generations), len(fitness))
    generations = generations[:temp_min]
    fitness = fitness[:temp_min]

generations = generations[0:400]
fitness = fitness[0:400]


# ax.plot(generations, fitness, label='Genetic Logistic Regression')

# %%
def string_to_dic(str_1):
    import json
    str_1 = str_1.replace("\'", "\"")
    res_1 = json.loads(str_1)
    return res_1


path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multi_label_gcn_comparision_down_sample_04291140.txt"
file1 = open(path, 'r')
Lines = file1.readlines()
# Strips the newline character
cora_comparision_only_train_gcn_test_fitness = []
cora_comparision_only_train_gcn_test_fitness_mean = []

for line in Lines:
    if line.startswith('comparison:'):
        temp_report = line.split('\n')[0].split('comparison:')[1]
        temp_report = string_to_dic(temp_report)
        cora_comparision_only_train_gcn_test_fitness.append((temp_report['1']['f1-score'] + temp_report['1']['f1-score'])/2)

# print("cora_comparision_only_train_gcn_test_fitness: {}".format(cora_comparision_only_train_gcn_test_fitness[-1]))
for i in range(len(cora_comparision_only_train_gcn_test_fitness)//10):
    temp_mean = np.array(cora_comparision_only_train_gcn_test_fitness[i:i+10]).mean()
    cora_comparision_only_train_gcn_test_fitness_mean.append(temp_mean)
cora_comparision_only_train_gcn_test_fitness_mean.sort()

path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multi_label_gcn_without_down_sample_04291212.txt"
file1 = open(path, 'r')
Lines = file1.readlines()
# Strips the newline character
cora_comparision_gcn_without_down_sample = []
cora_comparision_gcn_without_down_sample_mean = []

for line in Lines:
    if line.startswith('comparison:'):
        temp_report = line.split('\n')[0].split('comparison:')[1]
        temp_report = string_to_dic(temp_report)
        cora_comparision_gcn_without_down_sample.append((temp_report['1']['f1-score'] + temp_report['1']['f1-score'])/2)

# print("cora_comparision_only_train_gcn_test_fitness: {}".format(cora_comparision_only_train_gcn_test_fitness[-1]))
for i in range(len(cora_comparision_gcn_without_down_sample)//10):
    temp_mean = np.array(cora_comparision_gcn_without_down_sample[i:i+10]).mean()
    cora_comparision_gcn_without_down_sample_mean.append(temp_mean)
cora_comparision_gcn_without_down_sample_mean.sort()




path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multi_class_f1_half_train_GCN_04251736.txt"
file1 = open(path, 'r')
Lines = file1.readlines()
# Strips the newline character
cora_genetic_only_train_gcn_test_fitness = []
cora_genetic_only_train_gcn_train_fitness = []
for line in Lines:
    if line.startswith('test:'):
        temp_report = line.split('\n')[0].split('test:')[1]
        temp_report = string_to_dic(temp_report)
        if np.array(temp_report['total']['f1-score']).mean() > 0.7:
            print("{}, {}, {},accuracy:{}".format(np.array(temp_report['total']['f1-score']).mean(),
                                                  np.array(temp_report['total']['recall']).mean(),
                                                  np.array(temp_report['total']['precision']).mean(),
                                                  np.array(temp_report['accuracy']).mean()))
        cora_genetic_only_train_gcn_test_fitness.append(np.array(temp_report['total']['f1-score']).mean())
    if line.startswith('Fitness of the best solution :'):
        temp_report = float(line.split('\n')[0].split(':')[1])
        cora_genetic_only_train_gcn_train_fitness.append(temp_report)

cora_genetic_only_train_gcn_test_fitness_index = [i for i in range(len(cora_genetic_only_train_gcn_test_fitness))]

# %%

gcn_distribution.sort()
gcn_distribution_index = [i for i in range(len(gcn_distribution))]

ax.plot(gcn_distribution_index, gcn_distribution, label='gcn_distribution')

ax.plot(gcn_genetic_index, gcn_genetic_fitness, label='GCN Genetic')

# ax.plot(random_fitness_list_index, random_fitness_list, label='random under sample Logistic Regression')
## GCN
ax.plot([i for i in range(400)], [0.6861] * 400, label='GCN')

ax.plot([i for i in range(400)], [0.6742] * 400, label='under sample GCN')
cora_genetic_only_train_gcn_test_fitness.sort()
ax.plot(cora_genetic_only_train_gcn_test_fitness_index, cora_genetic_only_train_gcn_test_fitness, label='GCN Genetic New')
# cora_genetic_only_train_gcn_train_fitness_index = [i for i in range(len(cora_genetic_only_train_gcn_train_fitness))]

## DR-GCN
# ax.plot([i for i in range(400)], [(0.43 + 0.85) / 2] * 400, label='DR-GCN')

# ax.plot([i for i in range(400)], [(0.66 + 0.48) / 2] * 400, label='SMOTE Logistic Regression')

plt.xlabel('Generations')
plt.ylabel('Minority Classes F1 score')

# giving a title to my graph
plt.plot([], [], ' ', label="max fitness: {}".format(fitness[-1]))
plt.title("Cora")
plt.legend()

# function to show the plot
# plt.savefig('pic/evolution/Pubmed_EA_fitness_generations_{}.png'.format(len(generations)), dpi=300)
plt.show()
