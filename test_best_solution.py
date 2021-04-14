# %%
import numpy as np


best_population_path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multilabel_recall_04081338_best_solution.txt"
best_population_numpy_path = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_multilabel_recall_04081338_best_solution_numpy.txt"


def best_solution_str_to_int():
    # initial_population = np.loadtxt(fname=best_population_path, dtype=int, delimiter=' ')
    file1 = open(best_population_path, 'r')
    Lines = file1.readlines()

    solution = []
    for line in Lines:
        solution_str = line.split()
        for i in solution_str:
            solution.append(int(i))

    solution = np.array(solution).reshape(1, -1)
    np.savetxt(fname=best_population_numpy_path, fmt='%i', X=solution, delimiter=',')



