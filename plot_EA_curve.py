import matplotlib.pyplot as plt

# plotting the points


file1 = open('/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multi_label_gcn_genetic_0408_1542.txt',
             'r')
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

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(generations, fitness)
plt.xlabel('Generations')
plt.ylabel('Fitness')

# giving a title to my graph
plt.plot([], [], ' ', label="max fitness: {}".format(fitness[-1]))
plt.title("Cora optimize Recall")
plt.legend()

# function to show the plot
# plt.savefig('pic/evolution/Pubmed_EA_fitness_generations_{}.png'.format(len(generations)), dpi=300)
plt.show()
