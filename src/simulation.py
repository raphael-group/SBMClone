### Functions for generating simulated data ###
# Author: Matt Myers 

import numpy as np
from scipy import sparse 
from datetime import datetime



def generate_toy(n_cells = 100, n_muts = 1000, coverage=0.05, prop_clonal=0.10, part1_cell_fraction =0.5, part1_mut_fraction=0.5, seed=0, verbose=False, matrix_only = False):
    """
    Generate a simulated matrix that contains two subpopulations and sample it uniformly with block probability 'coverage'.
    """

    n_clonal = int(n_muts * prop_clonal)
    n_cells_part1 = int(part1_cell_fraction * n_cells)
    last_mut_part1 = int(part1_mut_fraction * n_muts * (1 - prop_clonal)) + n_clonal
    
    my_design = np.array([[1, 1, 0], [1, 0, 1]])
    row_thresholds = [n_cells_part1, n_cells]
    col_thresholds = [n_clonal, last_mut_part1, n_muts]

    M, cellsets, snvsets = generate_toy_design(my_design, row_thresholds, col_thresholds, coverage=coverage, seed=seed)
    if matrix_only:
        return M
    else:
        return M, cellsets, snvsets


# construct a block matrix with 1's in blocks indicated by design, and dimension indicate by rowdims and coldims
all_matrices = {} # memoize generated matrices to avoid re-generating matrices
def generate_toy_design(design, rowdims, coldims, coverage = 0.05, seed = 0):
    mblocks, nblocks = design.shape
    assert mblocks > 0
    assert nblocks > 0
    assert mblocks == len(rowdims)
    assert nblocks == len(coldims)
    assert coverage > 0
    
    rowdims = np.array(rowdims)
    coldims = np.array(coldims)

    # If your application requires generating the same matrix many times, uncomment this section and the line preceding this function.
    my_key = (design.tostring(), rowdims.tostring(), coldims.tostring(), coverage, seed) 
    if my_key in all_matrices:
        print("--- USING MEMOIZED MATRIX ---")
        return all_matrices[my_key]

    np.random.seed(seed)

    rboundaries = np.concatenate(([0], rowdims))
    cboundaries = np.concatenate(([0], coldims))
    
    rvalues = [rboundaries[i + 1] - rboundaries[i] for i in range(len(rboundaries) - 1)]
    cvalues = [cboundaries[i + 1] - cboundaries[i] for i in range(len(cboundaries) - 1)]
     
    # generate blocks
    blocks = [[np.random.binomial(1, design[i, j] * coverage, (rvalues[i], cvalues[j])) 
               if design[i, j] > 0 else np.zeros((rvalues[i], cvalues[j])) for j in range(nblocks)] for i in range(mblocks)]
    matrix = np.block(blocks)
        
    # Construct the true partition
    row_starts = np.append([0], rowdims[:-1])
    true_cellsets = [set([a for a in range(row_starts[i], rowdims[i])]) for i in range(len(rowdims))]
    col_starts = np.append([0], coldims[:-1])
    true_snvsets =  [set([a for a in range(col_starts[i], coldims[i])]) for i in range(len(coldims))]

    M = sparse.csr_matrix(matrix)
    all_matrices[my_key] = M, true_cellsets, true_snvsets
    
    return M, true_cellsets, true_snvsets

def generate_treesim(design, row_thresholds, col_thresholds, true_first_split = [], cov = 0.01, seed = 0, max_cluster_size = 2, version = 'impute'):
    """
    Generate a tree-structured simulated block matrix using the given parameters.
    """
    mblocks = len(row_thresholds)
    nblocks = len(col_thresholds)
    if len(true_first_split) > 0:
        assert len(true_first_split) == 2
        assert sum([len(a) for a in true_first_split]) == len(row_thresholds)
        assert all(a < mblocks for b in true_first_split for a in b)
        assert all(a >= 0 for b in true_first_split for a in b)
    else:
        true_first_split = [[0,1], list(range(2, 8))]
    
    print(datetime.now(), "Constructing matrix")
    M0, true_csets, true_ssets = generate_toy_design(design, row_thresholds, col_thresholds, coverage = cov, seed = seed)
    print(datetime.now(), "Merging cells within clones")

    if version == 'supernodes':
        M1, my_row_thresholds = merge_perfect(M0, true_csets, max_cluster_size = max_cluster_size, seed = seed, version = version)
        my_row_values = [(my_row_thresholds[i] - my_row_thresholds[i-1] if i > 0 else my_row_thresholds[i]) for i in range(len(my_row_thresholds))]        
        true_labels = [[i] * my_row_values[i] for i in range(len(my_row_values))]
        true_labels = np.concatenate(true_labels)
        
        return M1, true_labels, true_ssets
        
    else:
        M1 = merge_perfect(M0, true_csets, max_cluster_size = max_cluster_size, seed = seed, version = version)

        snv_split = [[8], [0, 1, 9], list(range(2, 8)) + list(range(10, 15))]
        cell_part = [set().union(*[true_csets[a] for a in true_first_split[i]]) for i in range(2)]
        snv_part = [set().union(*[true_ssets[a] for a in snv_split[i]]) for i in range(3)]
        
        return M1, cell_part + snv_part

def generate_basic_instances(ncells, nsnvs, cov, pc, ps, pcl, n_instances = 10, matrix_only = False):
    """ Generate n_instances simple block matrices  (2 cell/row blocks, 3 mutation/column blocks)"""
    # these depend on pcl (proportion of clonal SNVs) and prop
    clonal = int(pcl * nsnvs)
    clone1_muts = int(ps * nsnvs * (1 - pcl))
    clone2_muts = nsnvs - clonal - clone1_muts

    print(datetime.now(), "Started generating instances")
    instances = [generate_toy(n_cells = ncells, n_muts = nsnvs, coverage=cov, prop_clonal=pcl,
                                part1_cell_fraction = pc, part1_mut_fraction= ps, seed=sd, 
                                matrix_only = matrix_only) for sd in range(n_instances)]

    print(datetime.now(), "Done generating instances")

    return instances

def randomly_cluster(elements, max_cluster_size = 2, seed = 0, balanced = False):
    elements = list(elements)
    n_elements = len(elements)
    if max_cluster_size >= n_elements:
        return [elements]
        
    np.random.seed(seed)
    np.random.shuffle(elements)

    if balanced:
        # get as many clusters as possible with maximal size
        max_n_clusters = n_elements / max_cluster_size
        if n_elements % max_cluster_size > 0:
            max_n_clusters += 1
        cluster_size = n_elements / max_n_clusters
    else:
        # get the largest clusters possible even if 1 cluster is left tiny
        cluster_size = max_cluster_size
    
    clusters = []
    curr = 0
    while curr + cluster_size <= n_elements:
        clusters.append(elements[curr:curr + cluster_size])
        curr += cluster_size
    if curr < n_elements:
        clusters.append(elements[curr:n_elements])
    
    return clusters

def merge_perfect(M, clone_rows, max_cluster_size = 2, seed = 0, balanced = False, version = 'impute'):
    assert version == 'impute' or version == 'supernodes'
    if sparse.issparse(M):
        M = np.copy(M.todense())
    else:
        M = np.copy(M)
    
    # for each clone, cluster cells arbitrarily within this clone
    clusters = {}
    last_row_perclone = []
    i = 0
    for rows in clone_rows:
        # rows is a list of the row indexes corresponding to this clone        
        for cl in randomly_cluster(rows, max_cluster_size, seed, balanced = balanced):
            assert len(cl) <= max_cluster_size
            clusters[i] = cl
            i += 1
        last_row_perclone.append(i)

    if version == 'impute':
        # impute SNVs across rows within each cluster
        for idx, cl in list(clusters.items()):
            rows_idx = np.array(list(cl))
            #print cl
            #print rows_idx
            cols_idx = np.nonzero(M[rows_idx])[1]
            M[np.repeat(rows_idx, len(cols_idx)), np.tile(cols_idx, len(rows_idx))] = 1
        return sparse.csr_matrix(M)
    elif version == 'supernodes':
        cl_idx = 0
        rows = []
        cols = []
        data = []
        for _, cl in list(clusters.items()):
            rows_idx = np.array(list(cl))
            cols_idx = np.argwhere(np.any(M[rows_idx], axis = 0)).flatten()

            rows.extend([cl_idx] * len(cols_idx))
            cols.extend(list(cols_idx))
            data.extend([1] * len(cols_idx))
            cl_idx += 1
        return sparse.csr_matrix((data, (rows, cols)), shape = (cl_idx, M.shape[1])), last_row_perclone

########## Functions to generate matrices with explicit 0-entries for methods that take a ternary matrix ##########

# sample 0-entries as well for BnpC, SCITE, and SCG
def generate_toy_with0s(n_cells = 100, n_muts = 1000, coverage=0.05, prop_clonal=0.10, part1_cell_fraction =0.5, part1_mut_fraction=0.5, seed=0, verbose=False, matrix_only = False):
    n_clonal = int(n_muts * prop_clonal)
    n_cells_part1 = int(part1_cell_fraction * n_cells)
    last_mut_part1 = int(part1_mut_fraction * n_muts * (1 - prop_clonal)) + n_clonal
    
    my_design = np.array([[1, 1, 0], [1, 0, 1]])
    row_thresholds = [n_cells_part1, n_cells]
    col_thresholds = [n_clonal, last_mut_part1, n_muts]

    M, cellsets, snvsets = generate_toy_design_with0s(my_design, row_thresholds, col_thresholds, coverage=coverage, seed=seed)
    if matrix_only:
        return M
    else:
        return M, cellsets, snvsets

def generate_toy_design_with0s(design, rowdims, coldims, coverage = 0.05, seed = 0):
    mblocks, nblocks = design.shape
    assert mblocks > 0
    assert nblocks > 0
    assert mblocks == len(rowdims)
    assert nblocks == len(coldims)
    assert coverage > 0
    
    rowdims = np.array(rowdims)
    coldims = np.array(coldims)

    np.random.seed(seed)

    rboundaries = np.concatenate(([0], rowdims))
    cboundaries = np.concatenate(([0], coldims))
    
    rvalues = [rboundaries[i + 1] - rboundaries[i] for i in range(len(rboundaries) - 1)]
    cvalues = [cboundaries[i + 1] - cboundaries[i] for i in range(len(cboundaries) - 1)]
     
    # generate blocks
    blocks = [[generate_block_with0s(design[i, j] * coverage, (rvalues[i], cvalues[j])) 
               if design[i, j] > 0 else np.zeros((rvalues[i], cvalues[j])) for j in range(nblocks)] for i in range(mblocks)]
    matrix = np.block(blocks)
        
    # Construct the true partition
    row_starts = np.append([0], rowdims[:-1])
    true_cellsets = [set([a for a in range(row_starts[i], rowdims[i])]) for i in range(len(rowdims))]
    col_starts = np.append([0], coldims[:-1])
    true_snvsets =  [set([a for a in range(col_starts[i], coldims[i])]) for i in range(len(coldims))]

    M = sparse.csr_matrix(matrix)
    
    return M, true_cellsets, true_snvsets

def generate_block_with0s(s, shape):
    # sample 2 * cov entries
    my_block = np.random.binomial(1, 2 * s, shape) 
    
    # flip each entry to -1 with probability 0.5
    observed_0s = -2 * np.random.binomial(1, 0.5, shape) + 1

    return my_block * observed_0s

def generate_treesim_with0s(design, row_thresholds, col_thresholds, true_first_split = [], cov = 0.01, seed = 0, max_cluster_size = 2, version = 'impute'):
    """
    Generate a tree-structured simulated block matrix using the given parameters.
    """
    mblocks = len(row_thresholds)
    nblocks = len(col_thresholds)
    if len(true_first_split) > 0:
        assert len(true_first_split) == 2
        assert sum([len(a) for a in true_first_split]) == len(row_thresholds)
        assert all(a < mblocks for b in true_first_split for a in b)
        assert all(a >= 0 for b in true_first_split for a in b)
    else:
        true_first_split = [[0,1], list(range(2, 8))]
    
    print(datetime.now(), "Constructing matrix")
    M0, true_csets, true_ssets = generate_toy_design_with0s(design, row_thresholds, col_thresholds, coverage = cov, seed = seed)
    print(datetime.now(), "Merging cells within clones")

    if version == 'supernodes':
        M1, my_row_thresholds = merge_perfect_with0s(M0, true_csets, max_cluster_size = max_cluster_size, seed = seed, version = version)
        my_row_values = [(my_row_thresholds[i] - my_row_thresholds[i-1] if i > 0 else my_row_thresholds[i]) for i in range(len(my_row_thresholds))]        
        true_labels = [[i] * my_row_values[i] for i in range(len(my_row_values))]
        true_labels = np.concatenate(true_labels)
        
        return M1, true_labels, true_ssets
        
    else:
        M1 = merge_perfect_with0s(M0, true_csets, max_cluster_size = max_cluster_size, seed = seed, version = version)

        snv_split = [[8], [0, 1, 9], list(range(2, 8)) + list(range(10, 15))]
        cell_part = [set().union(*[true_csets[a] for a in true_first_split[i]]) for i in range(2)]
        snv_part = [set().union(*[true_ssets[a] for a in snv_split[i]]) for i in range(3)]
        
        return M1, cell_part + snv_part

def merge_perfect_with0s(M, clone_rows, max_cluster_size = 2, seed = 0, balanced = False, version = 'impute'):
    assert version == 'impute' or version == 'supernodes'
    if sparse.issparse(M):
        M = np.copy(M.todense())
    else:
        M = np.copy(M)
    
    # for each clone, cluster cells arbitrarily within this clone
    clusters = {}
    last_row_perclone = []
    i = 0
    for rows in clone_rows:
        # rows is a list of the row indexes corresponding to this clone        
        for cl in randomly_cluster(rows, max_cluster_size, seed, balanced = balanced):
            assert len(cl) <= max_cluster_size
            clusters[i] = cl
            i += 1
        last_row_perclone.append(i)

    if version == 'impute':
        # impute SNVs across rows within each cluster        
        for idx, cl in list(clusters.items()):
            rows_idx = np.array(list(cl))

            positive_cols_idx = np.argwhere(np.any(M[rows_idx] > 0, axis = 0)).flatten()
            negative_cols_idx = [j for j in np.argwhere(np.any(M[rows_idx] < 0, axis = 0)).flatten() if j not in positive_cols_idx]
            M[np.repeat(rows_idx, len(positive_cols_idx)), np.tile(positive_cols_idx, len(rows_idx))] = 1
            M[np.repeat(rows_idx, len(negative_cols_idx)), np.tile(negative_cols_idx, len(rows_idx))] = -1

            return sparse.csr_matrix(M)
    elif version == 'supernodes':
        cl_idx = 0
        rows = []
        cols = []
        data = []
        for _, cl in list(clusters.items()):
            rows_idx = np.array(list(cl))
            positive_cols_idx = np.argwhere(np.any(M[rows_idx] > 0, axis = 0)).flatten()
            # only include negative entries (observed absence) if they don't coincide with positive entries (observed presence)
            negative_cols_idx = [j for j in np.argwhere(np.any(M[rows_idx] < 0, axis = 0)).flatten() if j not in positive_cols_idx]
            
            rows.extend([cl_idx] * len(positive_cols_idx))
            cols.extend(list(positive_cols_idx))
            data.extend([1] * len(positive_cols_idx))
            
            rows.extend([cl_idx] * len(negative_cols_idx))
            cols.extend(list(negative_cols_idx))
            data.extend([-1] * len(negative_cols_idx))
            
            cl_idx += 1
        return sparse.csr_matrix((data, (rows, cols)), shape = (cl_idx, M.shape[1])), last_row_perclone