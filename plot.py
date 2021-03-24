import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

under_sample_ratios = [round(i * 0.03 + 0.08, 2) for i in range(30)]
over_sample_ratios = [round(i * 0.05 + 0.08, 2) for i in range(30)]
# under_sample_ratios = [round(i * 0.002 + 0.9, 3) for i in range(50)]
# over_sample_ratios = [round(i * 0.02 + 0.9, 3) for i in range(50)]
temp_list = []
for under_sample_ratio in under_sample_ratios:
    for over_sample_ratio in over_sample_ratios:
        temp_list.append([under_sample_ratio, over_sample_ratio])
xpos = []
ypos = []
remove = []
count = 0

for temp in temp_list:
    if temp[0] > temp[1]:
        remove.append(count)
    xpos.append(temp[0])
    ypos.append(temp[1])
    count = count + 1
effect = [i for i in range(900) if i not in remove]

xpos = np.array(xpos)
ypos = np.array(ypos)

file1 = open('/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/cora_0.08_0.5.log', 'r')
Lines = file1.readlines()
count = 0
recall_0 = []
# Strips the newline character
for line in Lines:

    if line[0] == '[':
        print(line)
        recall_0.append(float(str.split(line, sep=', ')[6]))
        count += 1

recall_0 = np.array(recall_0)
print(len(recall_0))
test = ['red'] * 50 + ['green'] * 850

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = 0.03 * np.ones_like(zpos)
dy = 0.05 * np.ones_like(zpos)

dz = recall_0
for i in remove:
    dz[i] = 0
xpos
list_of_colors = ['lightgrey', 'lightgray', 'silver', 'darkgrey', 'darkgray', 'grey', 'gray', 'dimgrey', 'dimgray',
                  'black']
colors = []
for i in dz:

    if i < 0.78:
        colors.append(list_of_colors[0])
        continue
    if i < 0.80:
        colors.append(list_of_colors[1])
        continue
    if i < 0.82:
        colors.append(list_of_colors[2])
        continue
    if i < 0.84:
        colors.append(list_of_colors[3])
        continue
    if i < 0.86:
        colors.append(list_of_colors[4])
        continue
    if i < 0.88:
        colors.append(list_of_colors[5])
        continue
    if i < 0.89:
        colors.append(list_of_colors[6])
        continue
    if i < 0.90:
        colors.append(list_of_colors[7])
        continue
    if i < 0.91:
        colors.append(list_of_colors[8])
        continue
    else:
        colors.append(list_of_colors[9])
        continue
print(len(colors))

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=colors)
ax.set_title('F1score value of Cora by logistic regression')
plt.savefig('Cora_0.08_0.95_F1score.png', dpi=1000)
plt.show()
