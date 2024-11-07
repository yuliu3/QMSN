import copy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import mylib
import matplotlib.ticker as ticker

p_p = 0.04
# p_f = 0.98
p_d = 0.8
p_s = 0.9
p_i = 0.9  # 1 db loss
n_e = np.power(10, 3)

num_qpus = 32
num_qubits = 128

print('simulation setting:', f"\n # of QPUs={num_qpus}", f"\n # of qubits={num_qubits}")
# map qpus to the locations in the networks,
# e.g., qpu i is mapped to location numbers[i] in the network
numbers = np.arange(num_qpus)
np.random.shuffle(numbers)

# get QPU pairs requesting entanglements under the qft circuit
pairs_arr = mylib.get_qpu_pairs_qft(num_qpus, num_qubits)
num_SD_pairs = pairs_arr.shape[0]
num_samples = 100

list_p_f = list(np.arange(start=0, stop=11, step=1) * 0.05 + 0.5)
nums_slot_grid = np.zeros([num_SD_pairs, num_samples, len(list_p_f)])
nums_slot_circle = np.zeros([num_SD_pairs, num_samples, len(list_p_f)])
nums_slot_line = np.zeros([num_SD_pairs, num_samples, len(list_p_f)], dtype=np.longdouble)
nums_attempts_switch = np.zeros([num_SD_pairs, num_samples, len(list_p_f)])
nums_attempts_benes = np.zeros([num_SD_pairs, num_samples, len(list_p_f)])

for idx_SD_pairs in range(num_SD_pairs):

    input_pair_grid = [mylib.get_node_grid(pairs_arr[idx_SD_pairs, 0], numbers), mylib.get_node_grid(pairs_arr[idx_SD_pairs, 1], numbers)]

    list_routes_grid = mylib.get_routes_grid_nodes(num_qpus, input_pair_grid)

    input_pair = [numbers[pairs_arr[idx_SD_pairs, 0]], numbers[pairs_arr[idx_SD_pairs, 1]]]
    list_routes_circle = mylib.get_routes_circle_nodes(num_qpus, input_pair)

    list_routes_line = mylib.get_routes_line_nodes(num_qpus, input_pair)

    for idx in range(len(list_p_f)):
        p_f = list_p_f[idx]

        prob_static_grid_slot, _ = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_grid, n_e)
        nums_slot_grid[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_static_grid_slot, size=num_samples)

        prob_switch = mylib.prob_switching_net(p_p, p_f, p_d, p_i, num_qpus)
        nums_attempts_switch[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_switch, size=num_samples)

        prob_benes = mylib.prob_benes_net(p_p, p_f, p_d, p_i, num_qpus)
        nums_attempts_benes[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_benes, size=num_samples)

nums_attempts_switch = nums_attempts_switch.reshape(-1, len(list_p_f))
nums_attempts_benes = nums_attempts_benes.reshape(-1, len(list_p_f))
nums_attempts_grid = n_e*nums_slot_grid.reshape(-1, len(list_p_f))
nums_attempts_circle = n_e*nums_slot_circle.reshape(-1, len(list_p_f))
nums_attempts_line = n_e*nums_slot_line.reshape(-1, len(list_p_f))

chain0 = copy.deepcopy(nums_attempts_switch.mean(axis=0))*10**(-6)
chain1 = copy.deepcopy(nums_attempts_benes.mean(axis=0))*10**(-6)
chain2 = copy.deepcopy(nums_attempts_grid.mean(axis=0)) * 10 ** (-6)
chain3 = copy.deepcopy(nums_attempts_circle.mean(axis=0)) * 10 ** (-6)
chain4 = copy.deepcopy(nums_attempts_line.mean(axis=0)) * 10 ** (-6)

range0 = [chain0.min(), chain0.max(), chain0.max()-chain0.min()]
range1 = [chain1.min(), chain1.max(), chain1.max()-chain1.min()]
range2 = [chain2.min(), chain2.max(), chain2.max()-chain2.min()]
range3 = [chain3.min(), chain3.max(), chain3.max()-chain3.min()]
range4 = [chain4.min(), chain4.max(), chain4.max()-chain3.min()]

colors = ['skyblue', 'darkorange', 'mediumseagreen', 'firebrick', 'orchid']
fig, ax1 = plt.subplots()
l0, = ax1.plot(list_p_f, chain0, color=colors[0], marker='*', linestyle='-', markersize='10', linewidth='2')
l1, = ax1.plot(list_p_f, chain1, color=colors[1], marker='^', linestyle='-', markersize='10', linewidth='2')
l2, = ax1.plot(list_p_f, chain2, color=colors[2], marker='d', linestyle='-', markersize='10', linewidth='2')
ax1.set_yscale("log")
ax1.legend([l0, l1, l2], ["QMSN", "Benes", "Grid"], fontsize=16)
ax1.grid(linestyle='--')
ax1.set_ylabel('Average Time for SEG', size=20)
ax1.set_xlim([0.5-0.5*0.02, 1+0.5*0.02])
ax1_yticks = [10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
ax1.set_yticks(ax1_yticks)
ax1.set_yticklabels([f'e{int(np.log10(tick))}' for tick in ax1_yticks])
ax1.tick_params(axis='both', which='major', labelsize=13)
fig.text(0.5, 0.02, '$p_f$', ha='center', fontsize=16)
plt.savefig("./plots/fig3b.pdf", format="pdf")
plt.show()

input_pdf = './plots/fig3b.pdf'  # Replace with your input PDF file path
output_pdf = './plots/fig_17.pdf'  # Replace with your desired output PDF file path
mylib.crop_pdf_upper_margin(input_pdf, output_pdf)
