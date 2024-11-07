import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import mylib

p_p = 0.04
p_f = 0.98
p_d = 0.8
p_s = 0.9
p_i = 0.9  # 1 db loss
n_e = 1*np.power(10, 3)
gamma = 10 ** (-6)
num_samples = 50

list_n = list([3, 4, 5, 6, 7])
nums_slot_grid = [None for _ in range(len(list_n))]
nums_attempts_qmsn = [None for _ in range(len(list_n))]
nums_attempts_bene = [None for _ in range(len(list_n))]

for idx in range(len(list_n)):
    num_qpus = 2**list_n[idx]
    numbers = np.arange(num_qpus)
    np.random.shuffle(numbers)
    pairs_arr = mylib.get_qpu_pairs_qft(num_qpus, num_qubits=num_qpus*4)
    num_SD_pairs = pairs_arr.shape[0]
    nums_slot_grid[idx] = [None for _ in range(num_SD_pairs)]
    nums_attempts_qmsn[idx] = [None for _ in range(num_SD_pairs)]
    nums_attempts_bene[idx] = [None for _ in range(num_SD_pairs)]
    print(f'iter = 1/{len(list_n)}, # of QPUs = {num_qpus}, # of SD pairs = {num_SD_pairs}')

    for idx_SD_pairs in range(num_SD_pairs):
        input_pair_grid = [mylib.get_node_grid(pairs_arr[idx_SD_pairs, 0], numbers),
                           mylib.get_node_grid(pairs_arr[idx_SD_pairs, 1], numbers)]

        list_routes_grid = mylib.get_routes_grid_nodes(num_qpus, input_pair_grid)

        prob_static_grid_slot, p_e = mylib.prob_static_net_slot(p_p, p_f, p_d, p_s, list_routes_grid, n_e)
        nums_slot_grid[idx][idx_SD_pairs] = np.random.geometric(p=prob_static_grid_slot, size=num_samples)

        prob_switch = mylib.prob_switching_net(p_p, p_f, p_d, p_i, num_qpus)
        nums_attempts_qmsn[idx][idx_SD_pairs] = np.random.geometric(p=prob_switch, size=num_samples)

        prob_benes = mylib.prob_benes_net(p_p, p_f, p_d, p_i, num_qpus)
        nums_attempts_bene[idx][idx_SD_pairs] = np.random.geometric(p=prob_benes, size=num_samples)
    nums_slot_grid_row = np.concatenate([nums_slot_grid[idx][j].flatten() for j in range(num_SD_pairs)])
    nums_slot_grid[idx] = nums_slot_grid_row

    nums_attempts_switch_row = np.concatenate([nums_attempts_qmsn[idx][j].flatten() for j in range(num_SD_pairs)])
    nums_attempts_qmsn[idx] = nums_attempts_switch_row

    nums_attempts_benes_row = np.concatenate([nums_attempts_bene[idx][j].flatten() for j in range(num_SD_pairs)])
    nums_attempts_bene[idx] = nums_attempts_benes_row

time_grid = []
time_bene = []
time_qmsn = []

for idx in range(len(list_n)):
    time_grid.append(nums_slot_grid[idx].mean()*n_e * gamma)
    time_bene.append(nums_attempts_bene[idx].mean() * gamma)
    time_qmsn.append(nums_attempts_qmsn[idx].mean() * gamma)


colors = ['skyblue', 'darkorange', 'mediumseagreen', 'firebrick', 'orchid']
fig, ax1 = plt.subplots()
l0, = ax1.plot(list_n, time_qmsn, color=colors[0], marker='*', linestyle='-', markersize='10', linewidth='2')
l1, = ax1.plot(list_n, time_bene, color=colors[1], marker='^', linestyle='-', markersize='10', linewidth='2')
l2, = ax1.plot(list_n, time_grid, color=colors[2], marker='d', linestyle='-', markersize='10', linewidth='2')
ax1.set_ylabel('Average Time for SEG', size=20)
ax1.tick_params(axis='y', which='major', labelsize=13)
fig.text(0.5, 0.02, '$\log(N)$', ha='center', fontsize=16)
ax1.set_yscale("log")
ax1.grid(axis='both', linestyle='--')
ax1.set_xticks([3, 4, 5, 6, 7])
ax1.set_xlim([list_n[0]-(list_n[-1]-list_n[0])*0.02, list_n[-1]+(list_n[-1]-list_n[0])*0.02])
ax1.legend([l0, l1, l2], ["QMSN", "Benes", "Grid"], loc=0, fontsize=16)
ax1_yticks = [10**-2, 10**-1, 10**0, 10**1, 10**2]
ax1.set_yticks(ax1_yticks)
ax1.set_yticklabels([f'e{int(np.log10(tick))}' for tick in ax1_yticks])
ax1.tick_params(axis='both', which='major', labelsize=13)
plt.savefig("./plots/fig3d.pdf", format="pdf")
plt.show()

input_pdf = './plots/fig3d.pdf'  # Replace with your input PDF file path
output_pdf = './plots/fig_19.pdf'  # Replace with your desired output PDF file path
mylib.crop_pdf_upper_margin(input_pdf, output_pdf)
