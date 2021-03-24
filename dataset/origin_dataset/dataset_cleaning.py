# In[1]
# 19717个点 每个点有五百个feature tx为1000*500的矩阵 allx为18717*500的矩阵 合在一起为全部feature
# ind_citeseer_allx = cPickle.load(open("data_zhs/data/ind.citeseer.allx", 'rb'))
# ind_citeseer_ally = cPickle.load(open("data_zhs/data/ind.citeseer.ally", 'rb'))
from scipy import sparse as sp

import argparse
import _pickle as cPickle
import numpy as np

path_prefix = "/RuiqiZheng/undergrad_thesis/undergrad_thesis_code/dataset/origin_dataset/data/"
ind_pubmed_graph = cPickle.load(open(path_prefix + "ind.pubmed.graph", 'rb'))
ind_pubmed_ty = cPickle.load(open(path_prefix + "ind.pubmed.ty", 'rb'))
# In[]
with open(path_prefix + "ind.pubmed.tx", 'rb') as f:
    ind_pubmed_tx = cPickle.load(f, encoding='latin1')
# ind_pubmed_tx = cPickle.load(open((path_prefix + "ind.pubmed.tx")))

with open(path_prefix + "ind.pubmed.x", 'rb') as f:
    ind_pubmed_x = cPickle.load(f, encoding='latin1')
with open(path_prefix + "ind.pubmed.y", 'rb') as f:
    ind_pubmed_y = cPickle.load(f, encoding='latin1')
# 18717 * 500
with open(path_prefix + "ind.pubmed.allx", 'rb') as f:
    ind_pubmed_allx = cPickle.load(f, encoding='latin1')
with open(path_prefix + "ind.pubmed.ally", 'rb') as f:
    ind_pubmed_ally = cPickle.load(f, encoding='latin1')

ind_pubmed_test_index = np.genfromtxt(path_prefix + "ind.pubmed.test.index")
# ind_pubmed_x = cPickle.load(open(path_prefix + "ind.pubmed.x", 'rb'))
# ind_pubmed_y = cPickle.load(open(path_prefix + "ind.pubmed.y", 'rb'))
# # 18717 * 500
# ind_pubmed_allx = cPickle.load(open(path_prefix + "ind.pubmed.allx", 'rb'))
#
# ind_pubmed_ally = cPickle.load(open(path_prefix + "ind.pubmed.ally", 'rb'))
#
# ind_pubmed_test_index = np.genfromtxt(path_prefix + "ind.pubmed.test.index")

# In[2]
count = [0,0,0]
for i in ind_pubmed_ally:
    for j in [0,1,2]:
        if i[j] == 1:
            count[j] = count[j] + 1
print(count)

# In[]
ind_pubmed_test_index.sort()
degree_map = {}
# for a in range(0, 19717):
#     for b in ind_pubmed_test_index:
#         break
train_list = []
test_train_map = {}
train_map = {}
for a in range(0, 19717):
    test_train_map[a] = 1

for test_node in ind_pubmed_test_index:
    test_train_map[test_node] = 0

count = 0
for a in range(0, 19717):
    if (test_train_map[a] == 1):
        train_map[a] = count
        count = count + 1

# print count
# print train_map[19003]
# for edge in ind_citeseer_graph:
# degree_map[edge[0]] = degree_map[edge[0]] + 1
# degree_map[edge[1]] = degree_map[edge[1]] + 1
# print edge
# break
# print (degree_map[1])

# In[3]
first_class = 0
second_class = 0
third_class = 0
for nodes in ind_pubmed_ally:
    if nodes[0] == 1:
        first_class = first_class + 1
    if nodes[1] == 1:
        second_class = second_class + 1
    if nodes[2] == 1:
        third_class = third_class + 1
a = [1, 2, 3]
print(first_class)
print(first_class * 1.0 / (first_class + second_class + third_class))
print(second_class)
print(third_class)
print(first_class + second_class + third_class)
# In[]
# x选取特定列
import copy

print(ind_pubmed_ally[a])
ind_pubmed_graph_del = copy.deepcopy(ind_pubmed_graph)
low_degree_node = []
for number in range(0, 18717):
    if ind_pubmed_ally[number][0] == 1:
        if len(ind_pubmed_graph[number]) <= 4:
            low_degree_node.append(number)

# for number in range(0, 1000):
#     if ind_pubmed_ty[number][0] == 1:
#         if len(ind_pubmed_graph[number]) <= 1000000:
#             low_degree_node.append(number)
count = 0
for low_node in low_degree_node:
    count = count + 1
    for related_node in ind_pubmed_graph[low_node]:

        try:
            ind_pubmed_graph_del[related_node].remove(low_node)
        except ValueError:
            print("related_node", related_node, "low_node", low_node)
    try:
        del ind_pubmed_graph_del[low_node]
    except ValueError:
        print("_____________________low_node", low_node)
print("the length of low degree node list: ", len(low_degree_node))
# In[5]
without_low_degree_node = []
for a in range(0, 18717):
    without_low_degree_node.append(a)
for a in low_degree_node:
    without_low_degree_node.remove(a)
print(len(without_low_degree_node))
ind_pubmed_allx_delete = ind_pubmed_allx[without_low_degree_node]
ind_pubmed_ally_delete = ind_pubmed_ally[without_low_degree_node]
ind_pubmed_allx_delete_np = ind_pubmed_allx_delete.toarray()
new_train_label = []
count = 0
for a in range(0, len(ind_pubmed_ally_delete)):
    if ind_pubmed_ally_delete[a][0] == 1:
        new_train_label.append(0)
        count = count + 1
    if ind_pubmed_ally_delete[a][1] == 1:
        new_train_label.append(1)
    if ind_pubmed_ally_delete[a][2] == 1:
        new_train_label.append(2)

print("train case minority count", count, "percent: ", count * 1.0 / 15621)
# In[4]
# for a in ind_pubmed_graph:
#     print a
#     break
# a = {0:[1,2]}
# a[0].remove(1)
ind_pubmed_allx_delete_np_label = np.insert(ind_pubmed_allx_delete_np, 500, new_train_label, axis=1)
# print(low_degree_node[186])
# print ind_pubmed_graph[1229]

# In[5]
ind_pubmed_tx_np = ind_pubmed_tx.toarray()
new_test_label = []
count = 0
for a in range(0, len(ind_pubmed_ty)):
    if ind_pubmed_ty[a][0] == 1:
        new_test_label.append(1)
        count = count + 1
    else:
        new_test_label.append(0)
ind_pubmed_tx_np_label = np.insert(ind_pubmed_tx_np, 500, new_test_label, axis=1)
print("test case minority count", count)
# In[6] for test
low_degree_node_test = []
for number in range(0, 1000):
    if ind_pubmed_ty[number][0] == 1:
        if len(ind_pubmed_graph[number + 18717]) <= 4:
            low_degree_node_test.append(number + 18717)

print("the length of low_degree_node_test", len(low_degree_node_test))

# In[7]
ind_pubmed_allx_np = ind_pubmed_allx.toarray()

# for a in ind_pubmed_tx_np:
#     ind_pubmed_allx_np.append(a)
ind_pubmed_train_test = np.append(ind_pubmed_allx_np, ind_pubmed_tx_np, axis=0)

ind_pubmed_label_train_test = np.append(ind_pubmed_ally, ind_pubmed_ty, axis=0)
first_class = 0
second_class = 0
third_class = 0
for nodes in ind_pubmed_label_train_test:
    if nodes[0] == 1:
        first_class = first_class + 1
    if nodes[1] == 1:
        second_class = second_class + 1
    if nodes[2] == 1:
        third_class = third_class + 1
print(first_class)
print(first_class * 1.0 / (first_class + second_class + third_class))
print(second_class)
print(third_class)
print(first_class + second_class + third_class)
low_degree_node = []
for number in range(0, 19717):
    if ind_pubmed_label_train_test[number][0] == 1:
        if len(ind_pubmed_graph[number]) <= 4:
            low_degree_node.append(number)
print(len(low_degree_node),
      (first_class - len(low_degree_node)) * 1.0 / (first_class + second_class + third_class - len(low_degree_node)))

# In[9]
# 删除第一类3229个点 还剩4103-3229 占比0.05300824842309559 现有19717-3229(16488)个点 874个少数点

old_node_to_new_node = {}
count = 0
for a in range(0, 19717):
    if a in low_degree_node:
        old_node_to_new_node[a] = -1
    else:
        old_node_to_new_node[a] = count
        count = count + 1
print(count)

# In[10]
ind_pubmed_graph_new = {}
for a in ind_pubmed_graph:
    if a in low_degree_node:
        continue
    temp_list = []
    for b in ind_pubmed_graph[a]:
        if b in low_degree_node:
            continue
        temp_list.append(old_node_to_new_node[b])
    ind_pubmed_graph_new[old_node_to_new_node[a]] = temp_list

# In[11]
count = 0
ind_pubmed_graph_new_degree_map = {}
for a in ind_pubmed_graph_new:
    if len(ind_pubmed_graph_new[a]) != 0:
        ind_pubmed_graph_new_degree_map[a] = count
        count = count + 1
# In[11]
## delete degree equals zero
zero_degree_node = []
for a in ind_pubmed_graph_new:
    if len(ind_pubmed_graph_new[a]) == 0:
        zero_degree_node.append(a)
# In[11]
ind_pubmed_graph_new_degree = copy.deepcopy(ind_pubmed_graph_new)
ind_pubmed_graph_new_degree_index = {}
count = 0
for low_node in zero_degree_node:
    count = count + 1
    del ind_pubmed_graph_new_degree[low_node]

for a in ind_pubmed_graph_new_degree:
    temp_list = []
    for b in ind_pubmed_graph_new_degree[a]:
        temp_list.append(ind_pubmed_graph_new_degree_map[b])
    ind_pubmed_graph_new_degree_index[ind_pubmed_graph_new_degree_map[a]] = temp_list
    # for related_node in ind_pubmed_graph_new[low_node]:
    #
    #     try:
    #         ind_pubmed_graph_del[related_node].remove(low_node)
    #     except ValueError:
    #         print ("related_node", related_node, "low_node", low_node)
    # try:
    #     del ind_pubmed_graph_del[low_node]
    # except ValueError:
    #     print("_____________________low_node", low_node)
# print ("the length of low degree node list: ", len(low_degree_node))

# In[11]
without_0_degree = []
for a in range(0, 16488):
    without_0_degree.append(a)
for a in zero_degree_node:
    without_0_degree.remove(a)

# In[11]
feature_select_list = []
for a in range(0, 19717):
    feature_select_list.append(a)
for a in low_degree_node:
    feature_select_list.remove(a)

ind_pubmed_train_test_select = ind_pubmed_train_test[feature_select_list]

ind_pubmed_label_train_test_select = ind_pubmed_label_train_test[feature_select_list]
new_train_test_label = []
for a in ind_pubmed_label_train_test_select:
    if a[0] == 1:
        new_train_test_label.append(1)
    else:
        new_train_test_label.append(0)
print("length of new_train_test_label", len(new_train_test_label))
ind_pubmed_train_test_select_plus_label = np.insert(ind_pubmed_train_test_select, 500, new_train_test_label, axis=1)

ind_pubmed_train_test_select_plus_label_without_0_degree = ind_pubmed_train_test_select_plus_label[without_0_degree]
np.savetxt("data_zhs/pubmed/features.pubmed", ind_pubmed_train_test_select_plus_label_without_0_degree)

# In[13]
# test_savetxt = np. genfromtxt("data_zhs/pubmed/features.pubmed")
#  19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338
# 39308
ind_pubmed_graph_new_list = []
for a in ind_pubmed_graph_new_degree_index:
    for b in ind_pubmed_graph_new_degree_index[a]:
        temp_list = []
        if (a < b):
            temp_list.append(a)
            temp_list.append(b)
        else:
            temp_list.append(b)
            temp_list.append(a)
        if (temp_list not in ind_pubmed_graph_new_list):
            ind_pubmed_graph_new_list.append(temp_list)
print(len(ind_pubmed_graph_new_list))
# In[14]
np.savetxt("data_zhs/pubmed/edges.pubmed", ind_pubmed_graph_new_list, fmt='%d')
# ind_pubmed_graph_new
# for a in ind_pubmed_graph_new:
#     if len(ind_pubmed_graph_new[a]) == 0:
#         print(a)
# In[15]

train_node_index = []
count = 0
for a in range(0, 16452):
    if ind_pubmed_train_test_select_plus_label_without_0_degree[a][-1] == 1:
        count = count + 1
print(count)
# In[15]
train_node_index = []
count = 0
for a in range(0, 16452):
    if ind_pubmed_train_test_select_plus_label_without_0_degree[a][-1] == 1:
        train_node_index.append(a)
        count = count + 1
    if (count >= 691):
        break
count = 0
for a in range(0, 16452):
    if ind_pubmed_train_test_select_plus_label_without_0_degree[a][-1] == 0:
        train_node_index.append(a)
        count = count + 1
    if (count >= 12470):
        break

print(len(train_node_index))
# print(count*1.0/(19717-3229))
# print(count)
print(train_node_index)
test_node_index = []
for a in range(0, 16452):
    if a not in train_node_index:
        test_node_index.append(a)
print(len(test_node_index))

# 训练集699少数类 12491多数类 少数类占比 0.05299469294920394
# 测试集175少数类 3123多数类 少数类占比 0.05306246209824136

# In[13]
np.savetxt("data_zhs/pubmed/train.pubmed", train_node_index, delimiter="\n", fmt='%d')
np.savetxt("data_zhs/pubmed/test.pubmed", test_node_index, delimiter="\n", fmt='%d')
# test_savetxt = np.genfromtxt("data_zhs/pubmed/test.pubmed")
