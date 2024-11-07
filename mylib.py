import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import fitz  # PyMuPDF

def get_routes_grid(num_qpus):

    num_log = np.log2(num_qpus)
    # print(num_qpus, num_log)
    if num_log % 2 == 1:
        num1 = num_log//2
        # print(f'shape = {2**num1}*{2**(num1+1)}')
        G = nx.grid_2d_graph(2**int(num1), 2**int(num1+1))
    else:
        G = nx.grid_2d_graph(int(np.sqrt(num_qpus)), int(np.sqrt(num_qpus)))
    while 1:
        target = random.choice(list(G.nodes))
        source = random.choice(list(G.nodes))
        if not target == source:
            break
    # Find all edge disjoint paths from source to target
    paths = list(nx.edge_disjoint_paths(G, source, target))

    # Create a list to store the number of links in each path
    links_in_paths = [len(path) - 1 for path in paths]
    # print(links_in_paths)
    return links_in_paths


def get_routes_line(num_qpus):
    random_numbers = random.sample(range(num_qpus), 2)
    return [np.abs(random_numbers[0]-random_numbers[1])]


def get_routes_circle(num_qpus):
    random_numbers = random.sample(range(num_qpus), 2)
    num = np.abs(random_numbers[0]-random_numbers[1])
    return [num, num_qpus-num]


def prob_route_slot(prob_p, prob_f, prob_d, prob_s, link_len, num_e):

    p_a = np.longdouble(str(0.5 * prob_d * (prob_f ** 2) * (prob_p ** 2)))

    p_e = 1-np.power(1 - p_a, num_e)

    p = np.power(p_e, link_len)*np.power(prob_s, link_len - 1)

    return p, p_e


def prob_static_net_slot(prob_p, prob_f, prob_d, prob_s, list_routes, num_e):
    p_r = np.longdouble('1')
    for route_len in list_routes:
        p_1, p_e = prob_route_slot(prob_p, prob_f, prob_d, prob_s, route_len, num_e)
        p_r = p_r*(1-p_1)
    return 1-p_r, p_e


def prob_switching_net(prob_p, prob_f, prob_d, prob_i, num_qpus):
    num_stages = np.log2(num_qpus)
    p_sa = 0.5 * prob_d * (prob_f**2) * (prob_p**2) * np.power(prob_i, 2*num_stages)
    return p_sa


def prob_benes_net(prob_p, prob_f, prob_d, prob_i, num_qpus):
    num_stages = np.log2(num_qpus)
    p_sa = 0.5 * prob_d * (prob_f**2) * (prob_p**2) * np.power(prob_i, 4*(num_stages-1))
    return p_sa


def geometric_samples(p, size):
    """
    Generate a list of random samples from a geometric distribution.
    """
    prob = np.longdouble(str(p))
    denominator = np.log(1 - prob)

    arr = np.zeros([size], dtype=np.longdouble)
    arr_pmf = np.zeros([size], dtype=np.longdouble)
    arr_pmf[1:] = np.arange(start=size-1, stop=0.1, step=-1)/size
    arr[1:] = np.ceil(np.log(arr_pmf[1:]) / denominator)
    arr[0] = 1
    return arr


def get_qpu_pairs_qft(num_qpus, num_qubits):

    num_qubits_per_qpu = num_qubits/num_qpus

    # List to store pairs of qubits that have gates between them
    qubit_gates = []

    # Generate gates in the pattern of a QFT circuit (increasing distances)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            qubit_gates.append([i, j])

    # Step 2: Map qubits to QPUs using the given allocation scheme
    # Each QPU holds num_qubits_per_qpu qubits, and qubits i and num_qubit-i are on the same QPU

    qpu_mapping = {}
    num_pairs_per_qpu = int(num_qubits_per_qpu // 2)
    for i in range(num_qpus):
        for j in range(num_pairs_per_qpu):
            # Map the lower end of the range
            qpu_mapping[i * num_pairs_per_qpu + j] = i
            # Map the mirrored upper end of the range
            qpu_mapping[num_qubits - 1 - (i * num_pairs_per_qpu + j)] = i

    # Step 3: Compute the list of QPU pairs requesting remote gates
    qpu_pairs = []

    # List to hold pairs of QPUs that require remote gates
    for gate in qubit_gates:
        qubit1, qubit2 = gate
        qpu1 = qpu_mapping[qubit1]
        qpu2 = qpu_mapping[qubit2]

        # Check if a remote gate is required (i.e., qubits are on different QPUs)
        if qpu1 != qpu2:
            # Sort to maintain consistent ordering (e.g., [0, 1] instead of [1, 0])
            qpu_pair = sorted([qpu1, qpu2])
            qpu_pairs.append(qpu_pair)
    return np.array(qpu_pairs)


def parse_cx_lines(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    cx_lines = [line for line in lines if line.startswith('cx')]

    arr = np.empty((len(cx_lines), 2), dtype=int)
    for i, line in enumerate(cx_lines):
        # Extract numbers within brackets []
        numbers = [int(s) for s in re.findall(r'\[([^[\]]*)\]', line)]
        arr[i, :] = np.array(numbers)

    return arr


def get_routes_grid_nodes(num_qpus, pair):

    num_log = np.log2(num_qpus)
    # print(num_qpus, num_log)
    if num_log % 2 == 1:
        num1 = num_log//2
        # print(f'shape = {2**num1}*{2**(num1+1)}')
        G = nx.grid_2d_graph(2**int(num1), 2**int(num1+1))
    else:
        G = nx.grid_2d_graph(int(np.sqrt(num_qpus)), int(np.sqrt(num_qpus)))

    target = pair[0]
    source = pair[1]

    # Find all edge disjoint paths from source to target
    paths = list(nx.edge_disjoint_paths(G, source, target))

    # Create a list to store the number of links in each path
    links_in_paths = [len(path) - 1 for path in paths]
    # print(links_in_paths)
    return links_in_paths


def get_routes_line_nodes(num_qpus, pair):
    random_numbers = pair
    return [np.abs(random_numbers[0]-random_numbers[1])]


def get_routes_circle_nodes(num_qpus, pair):
    random_numbers = pair
    num = np.abs(random_numbers[0]-random_numbers[1])
    return [num, num_qpus-num]


def get_node_grid(idx, numbers):

    i = np.log2(len(numbers))
    j = int(i//2)
    k = int(i-j)
    grid = numbers.reshape((2**j, 2**k))
    # print('get_node_grid', len(numbers), 2**j, 2**k)
    # grid = numbers.reshape((int(len(numbers)**0.5), int(len(numbers)**0.5)))

    # Find the position of idx in the grid
    position = np.argwhere(grid == idx)[0]

    return tuple(position)


def crop_pdf_upper_margin(input_pdf, output_pdf):
    """
    Crop the upper margin of a PDF file based on the topmost content boundary.
    """
    # Open the PDF file
    doc = fitz.open(input_pdf)

    for page_num, page in enumerate(doc, start=1):
        # Get all content blocks (text, images, etc.)
        blocks = page.get_text("blocks")

        if blocks:

            # Find the topmost y0 value from all blocks (uppermost content)
            # max_y0 = max(block[1] for block in blocks)
            max_y0 = 307

            # Define the new crop rectangle, setting the upper boundary to max_y0
            rect = page.rect
            print(rect)
            crop_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, max_y0)

            # Apply the new crop rectangle to the page
            page.set_cropbox(crop_rect)
            page.set_mediabox(crop_rect)

            print(f"Page {page_num}: Cropped upper margin down to y={max_y0:.2f}")
        else:
            print(f"Page {page_num}: No content found, page left unchanged.")

    # Save the cropped PDF to a new file
    doc.save(output_pdf)
    print(f"\nCropped PDF saved as '{output_pdf}'")