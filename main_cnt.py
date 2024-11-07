import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import mylib

filename = 'cnt3-5_180.txt'  # replace with your file name
pairs_arr = mylib.parse_cx_lines(filename)
num_SD_pairs = pairs_arr.shape[0]
print(pairs_arr.shape[0])
numbers = np.arange(16)
np.random.shuffle(numbers)

p_p = 0.04
p_f = 0.98
p_d = 0.8
p_s = 0.9
p_i = 0.9  # 1 db loss
n_e = np.power(10, 3)

num_qpus = 16

num_samples = 100

nums_slot_grid = np.zeros([num_SD_pairs, num_samples])
nums_slot_circle = np.zeros([num_SD_pairs, num_samples])
nums_slot_line = np.zeros([num_SD_pairs, num_samples], dtype=np.longdouble)
nums_attempts_switch = np.zeros([num_SD_pairs, num_samples])
nums_attempts_benes = np.zeros([num_SD_pairs, num_samples])

list_prob_line = []

for idx_SD_pairs in range(pairs_arr.shape[0]):

    input_pair_grid = [mylib.get_node_grid(pairs_arr[idx_SD_pairs, 0], numbers), mylib.get_node_grid(pairs_arr[idx_SD_pairs, 1], numbers)]
    input_pair = [numbers[pairs_arr[idx_SD_pairs, 0]], numbers[pairs_arr[idx_SD_pairs, 1]]]

    list_routes_grid = mylib.get_routes_grid_nodes(num_qpus, input_pair_grid)
    prob_static_grid_slot, _ = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_grid, n_e)
    nums_slot_grid[idx_SD_pairs, :] = np.random.geometric(p=prob_static_grid_slot, size=num_samples)

    list_routes_circle = mylib.get_routes_circle_nodes(num_qpus, input_pair)
    prob_static_circle_slot, _ = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_circle, n_e)
    # print(prob_static_circle_slot)
    nums_slot_circle[idx_SD_pairs, :] = np.random.geometric(p=prob_static_circle_slot, size=num_samples)

    list_routes_line = mylib.get_routes_line_nodes(num_qpus, input_pair)
    prob_static_line_slot, _ = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_line, n_e)
    list_prob_line.append(prob_static_line_slot)
    nums_slot_line[idx_SD_pairs, :] = mylib.geometric_samples(p=prob_static_line_slot, size=num_samples)

    prob_switch = mylib.prob_switching_net(p_p, p_f, p_d, p_i, num_qpus)
    nums_attempts_switch[idx_SD_pairs, :] = np.random.geometric(p=prob_switch, size=num_samples)

    prob_benes = mylib.prob_benes_net(p_p, p_f, p_d, p_i, num_qpus)
    nums_attempts_benes[idx_SD_pairs, :] = np.random.geometric(p=prob_benes, size=num_samples)

nums_attempts_grid = n_e*nums_slot_grid.reshape(-1)
nums_attempts_circle = n_e*nums_slot_circle.reshape(-1)
nums_attempts_line = n_e*nums_slot_line.reshape(-1)
nums_attempts_switch = nums_attempts_switch.reshape(-1)
nums_attempts_benes = nums_attempts_benes.reshape(-1)

chain1 = copy.deepcopy(nums_attempts_grid)*10**(-6)
chain2 = copy.deepcopy(nums_attempts_circle)*10**(-6)
chain3 = copy.deepcopy(nums_attempts_line)*10**(-6)
chain0 = copy.deepcopy(nums_attempts_switch)*10**(-6)
chainb = copy.deepcopy(nums_attempts_benes)*10**(-6)

print(f"time mean = {format(chain0.mean(), '.2e'), format(chain1.mean(), '.2e'), format(chain2.mean(), '.2e'), format(chain3.mean(), '.2e')}")

data = [chain0, chainb, chain1, chain2, chain3]
data_mean= [chain0.mean(), chainb.mean(), chain1.mean(), chain2.mean(), chain3.mean()]
labels = ['QMSN', 'Benes', 'Grid', 'Ring', 'Line']
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# colors = ["royalblue", "forestgreen", "maroon", "goldenrod", "slategray"]

colors = ['skyblue', 'royalblue', 'lightgreen', 'sandybrown', 'plum']

# plt.rcParams['figure.figsize'] = [6, 6]
bars = plt.bar(labels, data_mean, color=colors, width=0.5, edgecolor='black', hatch="/")
plt.yscale("log")
plt.ylabel('Average Time for SEG', size=20)
plt.xticks(fontsize=20)
plt.xticks(fontsize=16)
plt.grid()
q01 = chain0.min()
q03 = chain0.max()
iqr0 = q03 - q01
q11 = chain1.min()
q13 = chain1.max()
iqr1 = q13 - q11
q21 = chain2.min()
q23 = chain2.max()
iqr2 = q23 - q21
q31 = chain3.min()
q33 = chain3.max()
iqr3 = q33 - q31
qb1 = chainb.min()
qb3 = chainb.max()
iqrb = qb3 - qb1
# Display the IQR as error bars on the plot
plt.errorbar([0], [q01 + iqr0/2], yerr=iqr0/2, color='firebrick', linewidth=1, capsize=8)
plt.errorbar([1], [qb1 + iqrb/2], yerr=iqrb/2, color='firebrick', linewidth=1, capsize=8)
plt.errorbar([2], [q11 + iqr1/2], yerr=iqr1/2, color='firebrick', linewidth=1, capsize=8)
plt.errorbar([3], [q21 + iqr2/2], yerr=iqr2/2, color='firebrick', linewidth=1, capsize=8)
plt.errorbar([4], [q31 + iqr3/2], yerr=iqr3/2, color='firebrick', linewidth=1, capsize=8)
plt.savefig("./plots/fig1b.pdf", format="pdf", bbox_inches='tight')
plt.show()
print(data_mean/data_mean[0])


# fig, ax = plt.subplots()
# bp = plt.boxplot(data, notch=False,  # notch shape
#                  vert=True,  # vertical box alignment
#                  patch_artist=True,  # fill with color
#                  labels=labels,
#                  showfliers=False)
# # fill with colors
# colors = ['skyblue', 'lightgreen', 'sandybrown', 'plum']
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
#
# # show plot
# plt.xlabel('Topologies', size=16)
# plt.ylabel('Entanglement Generation Time', size=16)
# plt.yscale("log")
# ax.set_xticklabels(labels, fontsize=12)
# plt.grid()
# # plt.savefig("./plots/fig5.pdf", format="pdf", bbox_inches='tight')
# plt.show()
