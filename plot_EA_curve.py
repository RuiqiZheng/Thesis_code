import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import json
import numpy as np
# path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/citeseer_multi_label_0_5_comparision_gcn_04301806.txt"
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
    if line.startswith(('test:')):

        str_1 = line.replace("\'", "\"")
        str_1 = str_1.split("test:")[1]
        temp_dic = json.loads(str_1)
        fitness.append(np.array(temp_dic['accuracy']).mean())
        if np.array(temp_dic['accuracy']).mean() > 0.573:
            print(line)

        # if (np.array(temp_dic['total']['f1-score']).mean()) > 0.4:
        #     max_dic = temp_dic
        #     print(line)

    count = count + 1
    # print(line)
    # if count == 1000:
    #     break

print(fitness)
print(np.array(fitness).mean())
print(np.array(fitness).std())
# %%

# plotting the points
path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/citeseer_multilabel_LR_04141648.txt"

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
import json
def plot_random_fitness(path):
    random_fitness = np.loadtxt(fname=path, delimiter=',')
    return random_fitness
random_fitness = []
path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/citeseer_random_LR_400*50_result.txt"

file1 = open(path, 'r')
Lines = file1.readlines()
count = 0

# Strips the newline character

for line in Lines:
    if line.startswith('{\'0\': {'):

        str_1 = line.replace("\'", "\"")
        temp_dic = json.loads(str_1)

        random_fitness.append(temp_dic['0']['f1-score'])
        if (temp_dic['0']['f1-score'] >= 0.39):
            print(temp_dic)


random_fitness = np.array(random_fitness)
random_fitness_list = []
temp_max = 0
for i in range(int(len(random_fitness)/50)):
    temp_random_fitness = random_fitness[i*50:(i+1)*50]
    temp_max = max(temp_random_fitness.max(), temp_max)
    random_fitness_list.append(temp_max)
# %%
# random_fitness= plot_random_fitness("/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/citeseer_random_LR_400*50_result.txt")
# random_fitness = random_fitness.reshape(1,-1)[0]
# random_fitness_list = []
# temp_max = 0.5791738745200625
# i = 0
# for i in range(int(len(random_fitness)/50)):
#     temp_random_fitness = random_fitness[i*50:(i+1)*50]
#     temp_max = max(temp_random_fitness.max(), temp_max)
#     random_fitness_list.append(temp_max)

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
ax.plot(generations, fitness,label='Genetic Logistic Regression')

ax.plot(random_fitness_list_index, random_fitness_list, label='Random Logistic Regression')

ax.plot([i for i in range(400)], [0.3300] * 400, label='SMOTE Logistic Regression')
ax.plot([i for i in range(400)], [0.3241] * 400, label='GCN')
plt.xlabel('Generations')
plt.ylabel('Fitness')

# giving a title to my graph
plt.plot([], [], ' ', label="max fitness: {}".format(fitness[-1]))
plt.title("Citeseer optimize F1score")
plt.legend()

# function to show the plot
# plt.savefig('pic/evolution/Pubmed_EA_fitness_generations_{}.png'.format(len(generations)), dpi=300)
plt.show()
