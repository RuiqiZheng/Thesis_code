import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(9, 9))
if len(generations) != len(fitness):
    print("length of generation: {}, length of fitness: {}".format(len(generations), len(fitness)))
    temp_min = min(len(generations), len(fitness))
    generations = generations[:temp_min]
    fitness = fitness[:temp_min]

generations = generations[0:400]
fitness = fitness[0:400]
ax.plot(generations, fitness, label='Genetic Logistic Regression')

ax.plot(random_fitness_list_index, random_fitness_list, label='random under sample Logistic Regression')
## GCN
ax.plot([i for i in range(400)], [0.6356556291390729] * 400, label='GCN')

## DR-GCN
ax.plot([i for i in range(400)], [(0.43 + 0.85) / 2] * 400, label='DR-GCN')

ax.plot([i for i in range(400)], [(0.66 + 0.48) / 2] * 400, label='SMOTE Logistic Regression')

plt.xlabel('Generations')
plt.ylabel('Minority Classes F1 score')

# giving a title to my graph
plt.plot([], [], ' ', label="max fitness: {}".format(fitness[-1]))
plt.title("Cora")
plt.legend()

# function to show the plot
# plt.savefig('pic/evolution/Pubmed_EA_fitness_generations_{}.png'.format(len(generations)), dpi=300)
plt.show()
