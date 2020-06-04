### General utility functions ###
# Author: Matt Myers 

import numpy as np
from scipy import sparse 
import graph_tool
import networkx as nx
from collections import Counter
import os
import matplotlib.pyplot as plt
from datetime import datetime

def log(msg):
    print(datetime.now().strftime("%H:%M:%S") + " " + msg)

def visualize_matrix(M, row_partition, col_partition, filename):
    """
    Visualize the matrix M using blocks encoded in row labels <row_partition> and column labels <col_partition> and write it to <filename>. 
    
    In the SBM inference, the row and column blocks are numbered together (rather than row blocks being 1...k and column blocks being 1...l). Setting <include_index> to True will include the internal block index of each row/column block in the visualization.
    """

    rho = density_matrix(M, row_partition, col_partition)
    assert len(rho) > 0
    assert len(rho[0]) > 0

    k = len(rho)
    l = len(rho[0])
    
    fig, ax = plt.subplots(dpi = 250)
    im = ax.imshow(rho, cmap = 'gray_r')
        
    ax.set_xticks(np.arange(l))
    ax.set_yticks(np.arange(k))

    ax.set_yticklabels(["Clone {}".format(i + 1) for i in range(k)])
    ax.set_xticklabels(["Mutation cluster {}".format(i + 1) for i in range(l)])

    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
    
    threshold = 0.2 * np.max(rho)
    for i in range(k):
        for j in range(l):
            text = ax.text(j, i, "{:.4f}".format(rho[i][j]),
                           ha="center", va="center", color="w" if  rho[i][j] > threshold else 'black')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(filename)

def density_matrix(M, row_part, col_part):
    """
    Given a sparse matrix M, row labels, and column labels, constructs a block matrix where each entry contains the proportion of 1-entries in the corresponding rows and columns.
    """
    m, n = M.shape
    if m <= 0 or n <= 0:
        raise ValueError("Matrix M has dimensions with 0 or negative value.")
    if m != len(row_part):
        raise ValueError("Row labels must be the same length as the number of rows in M.")
    if n != len(col_part):
        raise ValueError("Column labels must be the same length as the number of columns in M.")

    row_groups = Counter(row_part).keys()
    col_groups = Counter(col_part).keys()
    #print row_groups, col_groups
    
    row_part = np.array(row_part)
    col_part = np.array(col_part)
    
    row_idx = [np.where(row_part == a)[0] for a in row_groups]
    col_idx = [np.where(col_part == b)[0] for b in col_groups]
    #print [len(a) for a in row_idx]
    #print [len(b) for b in col_idx]
    
    density_matrix = [[np.sum(M[row_idx[i]][:, col_idx[j]]) / float(len(row_idx[i]) * len(col_idx[j])) for j in range(len(col_groups))] for i in range(len(row_groups))]
    return density_matrix

def read_matrix_file(fname, verbose = True):
    """
    Given a CSV file where each line contains the row and column indices of a single 1-entry, construsts a scipy CSR sparse matrix.
    """

    parse_entry = lambda x: (int(x[0]), int(x[1]))
    rows = []
    cols = []
    with open(fname, "r") as f:
        line_number = 1
        for line in f:
            i, j = parse_entry(line.strip().split(','))
            if i < 0 or j < 0:
                raise ValueError("Row and column indices must be non-negative (line {}).".format(line_number))
        
            rows.append(i)
            cols.append(j)
            line_number += 1
    data = [1] * len(rows)
    coo = sparse.coo_matrix((data, (rows, cols)))
    if verbose:
        log("Input matrix has {} rows, {} columns, and {} 1-entries.".format(coo.shape[0], coo.shape[1], coo.nnz))
    return coo.tocsr()

def write_matrix_file(M, fname):
    """
    Writes the sparse matrix M to a CSV file in the desired input format.
    """
    assert sparse.issparse(M)
    if not isinstance(M, sparse.coo_matrix):
        M = M.tocoo()

    with open(fname, 'w') as f:
        for pair in zip(M.row, M.col):
            f.write(','.join([str(a) for a in pair]) + os.linesep)

def blockmodel_to_labels(b, label2id, n_cells = None, n_muts = None, maxblocks = 100):
    """
    Converts the graph_tool blockmodel return objects to partition vectors for cells (rows) and mutations (columns).
    """
    if n_cells is None:
        n_cells = max([int(x[4:]) if x.startswith('cell') else 0 for x in label2id.keys()])
    if n_muts is None:
        n_muts = max([int(x[3:]) if x.startswith('mut') else 0 for x in label2id.keys()])
    assert n_cells > 0
    assert n_muts > 0
    
    
    cell_array = [(b[label2id['cell{}'.format(i)]] if 'cell{}'.format(i) in label2id else maxblocks + 1) for i in range(n_cells)]
    temp = sorted(list(Counter(cell_array).keys()))
    cell_idx_to_blocknum = {temp[i]:i + 1 for i in range(len(temp))}
    cell_array = [cell_idx_to_blocknum[a] for a in cell_array]
    
    mut_array = [(b[label2id['mut{}'.format(i)]] if 'mut{}'.format(i) in label2id else maxblocks + 1) for i in range(n_muts)]
    temp = sorted(list(Counter(mut_array).keys()))
    mut_idx_to_blocknum = {temp[i]:i + 1 for i in range(len(temp))}
    mut_array = [mut_idx_to_blocknum[a] for a in mut_array]
    return cell_array, mut_array

reverse_dict = lambda d:  {v:k for k,v in list(d.items())}

def construct_graph_graphtool(M):
    """
    Given an (unweighted, undirected) biadjacency matrix M, construct a graph_tool graph object with the corresponding edges.
    Returns the graph G as well as a mapping from vertices of G to cells (rows) and mutations (columns) in M. 
    """
    M = M.tocoo()
    G = graph_tool.Graph(directed = False)
    
    vtype = G.new_vertex_property("short")

    label2id = {}
    for i,j,value in zip(M.row, M.col, M.data):
        assert value == 1
        cell_key = 'cell{}'.format(i)
        mut_key = 'mut{}'.format(j)
        if cell_key in label2id:
            v = label2id[cell_key]
        else:
            v = G.add_vertex()
            label2id[cell_key] = int(v)
            vtype[v] = 1
            
        if mut_key in label2id:
            w = label2id[mut_key]
        else:
            w = G.add_vertex()
            label2id[mut_key] = int(w)
            vtype[w] = 2

        G.add_edge(v, w)  
    return G, label2id, vtype

def graph_to_csr(G):
    """
    Constructs the biadjacency matrix of a graph that is assumed to be bipartite.
    """
    cell_to_idx = {}
    mut_to_idx = {}
    cidx = 0
    midx = 0
    for node in G.nodes:
        if type(node) == str:
            cell_to_idx[node] = cidx
            cidx += 1
        else:
            mut_to_idx[node] = midx
            midx += 1
    
    rowind = []
    colind = []
    data = []

    for edge in G.edges:
        u = edge[0]
        v = edge[1]
        if type(u) == str:
            rowind.append(cell_to_idx[u])
            colind.append(mut_to_idx[v])
        else:
            rowind.append(cell_to_idx[v])
            colind.append(mut_to_idx[u])
        data.append(1)
        
    return sparse.csr_matrix((data, (rowind, colind)), shape=(cidx, midx)), cell_to_idx, mut_to_idx

    # adapted from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
def csr_delete(mat, indices, axis=0):
    """
    Delete rows or columns from a SciPy CSR matrix.

    Args:
        mat: matrix to be processed
        indices: indices of the rows/columns to be deleted
        axis: 0 for rows, 1 for columns
    Returns:
        matrix with designated rows/columns removed
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise TypeError("works only for CSR format -- use .tocsr() first")
    if not (axis == 0 or axis == 1):
        raise ValueError("axis must be 0 for rows or 1 for columns")
    if not isinstance(indices, list):
        indices = list(indices)
    mask = np.ones(mat.shape[axis], dtype=bool)
    mask[indices] = False
    if axis == 0:
        return mat[mask]
    else:
        assert axis == 1
        return mat[:,mask]

def sets_to_clusters(sets, idx_to_name):
    """
    Converts a set representation of a partition to a vector of object labels.
    """

    assert len(idx_to_name) == sum([len(a) for a in sets])
    
    arr = []
    for i in range(len(idx_to_name)):
        name = idx_to_name[i]
        cl = [j for j in range(len(sets)) if name in sets[j]]
        assert len(cl) == 1
        cl = cl[0]
        arr.append(cl)
        
    return arr