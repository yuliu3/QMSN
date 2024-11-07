import copy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import mylib
import matplotlib.ticker as ticker

# p_p = 0.06
p_f = 0.98
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
num_samples = 20

list_p_p = list(np.arange(start=0, stop=30, step=1)*0.005+0.01)
nums_slot_grid = np.zeros([num_SD_pairs, num_samples, len(list_p_p)])
nums_slot_circle = np.zeros([num_SD_pairs, num_samples, len(list_p_p)])
nums_slot_line = np.zeros([num_SD_pairs, num_samples, len(list_p_p)], dtype=np.longdouble)
nums_attempts_switch = np.zeros([num_SD_pairs, num_samples, len(list_p_p)])
nums_attempts_benes = np.zeros([num_SD_pairs, num_samples, len(list_p_p)])

for idx_SD_pairs in range(num_SD_pairs):

    # list_routes_grid = mylib.get_routes_grid(num_qpus)
    #
    # list_routes_circle = mylib.get_routes_circle(num_qpus)
    #
    # list_routes_line = mylib.get_routes_line(num_qpus)

    input_pair_grid = [mylib.get_node_grid(pairs_arr[idx_SD_pairs, 0], numbers), mylib.get_node_grid(pairs_arr[idx_SD_pairs, 1], numbers)]

    list_routes_grid = mylib.get_routes_grid_nodes(num_qpus, input_pair_grid)

    input_pair = [numbers[pairs_arr[idx_SD_pairs, 0]], numbers[pairs_arr[idx_SD_pairs, 1]]]
    list_routes_circle = mylib.get_routes_circle_nodes(num_qpus, input_pair)

    list_routes_line = mylib.get_routes_line_nodes(num_qpus, input_pair)

    for idx in range(len(list_p_p)):
        p_p = list_p_p[idx]
        if p_p >= 0.02:
            prob_static_grid_slot, p_e = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_grid, n_e)
            nums_slot_grid[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_static_grid_slot, size=num_samples)

        if p_p >= 0.03:
            prob_static_circle_slot, p_e = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_circle, n_e)
            nums_slot_circle[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_static_circle_slot, size=num_samples)

        if p_p>= 0.04:
            prob_static_line_slot, p_e = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_line, n_e)
            nums_slot_line[idx_SD_pairs, :, idx] = mylib.geometric_samples(p=prob_static_line_slot, size=num_samples)

        prob_switch = mylib.prob_switching_net(p_p, p_f, p_d, p_i, num_qpus)
        nums_attempts_switch[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_switch, size=num_samples)
        prob_benes = mylib.prob_benes_net(p_p, p_f, p_d, p_i, num_qpus)
        nums_attempts_benes[idx_SD_pairs, :, idx] = np.random.geometric(p=prob_benes, size=num_samples)

nums_attempts_switch = nums_attempts_switch.reshape(-1, len(list_p_p))
nums_attempts_benes = nums_attempts_benes.reshape(-1, len(list_p_p))
nums_attempts_grid = n_e*nums_slot_grid.reshape(-1, len(list_p_p))
nums_attempts_circle = n_e*nums_slot_circle.reshape(-1, len(list_p_p))
nums_attempts_line = n_e*nums_slot_line.reshape(-1, len(list_p_p))

chain0 = copy.deepcopy(nums_attempts_switch.mean(axis=0))*10**(-6)
chain1 = copy.deepcopy(nums_attempts_benes.mean(axis=0))*10**(-6)
chain2 = copy.deepcopy(nums_attempts_grid.mean(axis=0)) * 10 ** (-6)
chain3 = copy.deepcopy(nums_attempts_circle.mean(axis=0)) * 10 ** (-6)
chain4 = copy.deepcopy(nums_attempts_line.mean(axis=0)) * 10 ** (-6)

# plot the data
colors = ['skyblue', 'darkorange', 'mediumseagreen', 'firebrick', 'orchid']
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.subplots_adjust(wspace=0.00)
ax2d = ax2.twinx()
l0, = ax1.plot(list_p_p[:8], chain0[:8], color=colors[0], marker='*', linestyle='-')
l1, = ax1.plot(list_p_p[:8], chain1[:8], color=colors[1], marker='^', linestyle='-')
l2, = ax1.plot(list_p_p[2:8], chain2[2:8], color=colors[2], marker='d', linestyle='-')
l3, = ax1.plot(list_p_p[4:8], chain3[4:8], color=colors[3], marker='+', linestyle='-')
l4, = ax1.plot(list_p_p[6:8], chain4[6:8], color=colors[4], marker='o', linestyle='-')
ax1.set_yscale("log")
ax1.legend([l0, l1, l2, l3, l4], ["QMSN", "Benes", "Grid", "Ring", "Line"], loc='upper left', fontsize=13)
ax1.spines['right'].set_visible(False)
ax1.set_ylim([0.001, np.power(10, 9)])
ax1_yticks = [10**-1, 10**1, 10**3, 10**5, 10**7]
ax1.set_yticks(ax1_yticks)
ax1.set_yticklabels([f'e{int(np.log10(tick))}' for tick in ax1_yticks])
ax1_xticks = [0.01, 0.02, 0.03, 0.04]
ax1.set_xticks(ax1_xticks)
ax1.set_xticklabels([f'{int(tick/0.01)}e-2' for tick in ax1_xticks])
# ax1_yticks = [0.1, np.power(10, 1), np.power(10, 3), np.power(10, 5), np.power(10, 7)]
# ax1.set_yticks(ax1_yticks)
ax1.grid(True, linestyle='--', alpha=0.7, which='both')
ax1.set_ylabel('Average Time for SEG', size=20)
ax1.tick_params(axis='both', which='major', labelsize=13)

l0, = ax2d.plot(list_p_p[8:], chain0[8:], color=colors[0], marker='*', linestyle='-')
l1, = ax2d.plot(list_p_p[8:], chain1[8:], color=colors[1], marker='^', linestyle='-')
l2, = ax2d.plot(list_p_p[8:], chain2[8:], color=colors[2], marker='d', linestyle='-')
ax2.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax2.set_yticks([])
ax2.set_xticks([.06, .08, .1, .12, .14], ['.06', '.08', '.10', '.12', '.14'])
ax2.tick_params(axis='x', which='major', labelsize=13)
ax2d.spines['left'].set_color('silver')
# ax2d.spines['left'].set_visible(False)
formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+'))
ax2d.set_yticks([0.002, 0.004, 0.006, 0.008, 0.01])
ax2d.yaxis.set_major_formatter(formatter)
ax2d.set_ylim([0, 0.012])
ax2d.legend([l0, l1, l2], ["QMSN", "Benes", "Grid"], fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7, which='both')
ax2d.grid(True, linestyle='--', alpha=0.7, which='both')
fig.text(0.5, 0.02, '$p_p$', ha='center', fontsize=16)
ax2d.tick_params(axis='both', which='major', labelsize=13)
# plt.savefig("./plots/fig3a.pdf", format="pdf")
plt.show()

input_pdf = './plots/fig3a.pdf'  # Replace with your input PDF file path
output_pdf = './plots/fig_16.pdf'  # Replace with your desired output PDF file path
mylib.crop_pdf_upper_margin(input_pdf, output_pdf)
