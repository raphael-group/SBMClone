### Functions used during the research project that culminated in SBMClone ###
# Author: Matt Myers 

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import graph_tool.inference.minimize as gt_min
import scipy
import numpy as np 
from datetime import datetime
from util import *
from simulation import generate_toy_design, merge_perfect

########## Functions to test SBMClone ##########
def test_graphtool_recursive(M, min_blocks = 4, max_blocks = 5, nested = True):
    """
    Given biadjacency matrix M, infers a stochastic block model fitting M with at most 5 blocks (first split),
    then divides the cells into sets accordingly and infers an SBM with at most 5 blocks from each induced submatrix.
    ARI is computed with reference to the 4085-cell tree-structured block matrix.
    """
    results = []

    row_values = [129, 1823, 72, 1686, 141, 39, 88, 107]
    true_labels = [[i] * row_values[i] for i in range(len(row_values))]
    true_labels = np.concatenate(true_labels)

    first_labels = [0] * 1952 + [1] * 2133
    first_ari, (cellpart, snvpart) = test_graphtool(M, first_labels, min_blocks, max_blocks, nested)

    print("First split ARI: {}".format(first_ari))

    inf_labels = np.array(cellpart)
    cell_sets = {i:np.argwhere(inf_labels == i).flatten() for i in range(max_blocks + 1)}

    print([len(a) for a in cell_sets.values()])

    results.append((true_labels, first_ari, cellpart, snvpart))
    # break M into submatrices by rows using cell clusters
    # For each submatrix, if the submatrix meets some condition,
    for cset in cell_sets.values():
        if len(cset) > 1:
            myM = M[cset]
            my_labels = [true_labels[a] for a in cset]
            
            next_ari, (cellpart, snvpart) = test_graphtool(myM, my_labels, min_blocks, max_blocks, nested)

            print("Recursive split ARI: {} (on {} cells)".format(next_ari, len(cset)))

            results.append((my_labels, next_ari, cellpart, snvpart))

    return results


def test_graphtool(M, true_labels, min_blocks = 4, max_blocks = None, nested = True):
    """
    Apply the SBM inference functions from graph_tool to the given biadjacency matrix M.
    Returns the ARI between the inferred partition and the partition [true_part[0], true_part[1:]]
    """
    assert min_blocks is None or isinstance(min_blocks, int)
    assert max_blocks is None or isinstance(max_blocks, int)
    m, n = M.shape
    G, label2id, vtype = construct_graph_graphtool(M)
    if nested:
        r = gt_min.minimize_nested_blockmodel_dl(G, B_min = min_blocks, B_max = max_blocks, state_args = {'clabel': vtype})
        b = r.get_bs()[0]
    else:
        r = gt_min.minimize_blockmodel_dl(G, B_min = min_blocks, B_max = max_blocks, state_args = {'clabel': vtype})
        b = r.get_blocks()
    
    id2label = reverse_dict(label2id)
    inf_labels = [b[label2id['cell{}'.format(i)]] if 'cell{}'.format(i) in label2id else 100 for i in range(m)]
    return adjusted_rand_score(inf_labels, true_labels)

def test_graphtool_oneshot(M, true_labels = None, min_blocks = 4, max_blocks = None, nested = True):
    """
    Apply the SBM inference functions from graph_tool to the given biadjacency matrix M.
    If true_labels is not given, assumes M is the empirical tree structure and uses the ground truth labeling of the 4085 cells.
    Returns the ARI between the inferred partition and the partition given by true_lables, as well as the inferred partition.
    """
    m, n = M.shape
    G, label2id, vtype = construct_graph_graphtool(M)
    if nested:
        r = gt_min.minimize_nested_blockmodel_dl(G, B_min = min_blocks, B_max = max_blocks, state_args = {'clabel': vtype})
        b = r.get_bs()[0]
    else:
        r = gt_min.minimize_blockmodel_dl(G, B_min = min_blocks, B_max = max_blocks, state_args = {'clabel': vtype})
        b = r.get_blocks()

    if true_labels is None:
        row_values = [129, 1823, 72, 1686, 141, 39, 88, 107]
        true_labels = [[i] * row_values[i] for i in range(len(row_values))]
        true_labels = np.concatenate(true_labels)

    cell_part, mut_part = blockmodel_to_labels(b, label2id, n_cells = m, n_muts = n)
    inf_labels = cell_part
    
    return adjusted_rand_score(true_labels, inf_labels), cell_part, mut_part

########## Functions to test naive approaches #########
def test_naive_rows_kmeans(M, true_part, k = 2):
    """
    Naive k-means approach used in the paper.
    """

    # Compute the sum for each row to use as a 1-dimensional representation of the row
    rowsums = np.sum(M, axis=1)
    row_idx_pairs = sorted(zip(rowsums, range(len(rowsums))), key=lambda x: x[0], reverse = True)
    row_idx = [x[1] for x in row_idx_pairs]
    
    # Cluster rows using kmeans
    pred_labels = KMeans(n_clusters = k, random_state = 0).fit_predict(rowsums.reshape(-1, 1))
    
    if len(true_part) == 2:
        true_labels = [(0 if i in true_part[0] else 1) for i in range(len(true_part[0]) + len(true_part[1]))]
    else:
        true_labels = true_part

    score = adjusted_rand_score(true_labels, pred_labels)
    return score


def test_naive_cols_kmeans(M, true_part):
    """
    Column/mutation-based k-means approach. Clusters columns/mutations into 3 sets using k-means on their sums, calls the set with largest sum clonal, and matches each row to the set of columns that it has the most 1-entries in.
    """

    # Compute column sums
    colsums = np.sum(M, axis=0)

    # Cluster columns using kmeans
    col_labels = KMeans(n_clusters = 3, random_state = 0).fit_predict(colsums.reshape(-1, 1))

    # Call the section with the largest average column sum clonal
    idx0 = []
    idx1 = []
    idx2 = []
    for idx in range(len(col_labels)):
        label = col_labels[idx]
        if label == 0:
            idx0.append(idx)
        elif label == 1:
            idx1.append(idx)
        elif label == 2:
            idx2.append(idx)
        else:
            print("uhoh")
    
    colsums = colsums.reshape(-1, 1)
    mean0 = np.mean(colsums[idx0])
    mean1 = np.mean(colsums[idx1])
    mean2 = np.mean(colsums[idx2])
    
    if mean0 > mean1:
        if mean0 > mean2:
            clonal_idx = idx0
            subclone1_idx = idx1
            subclone2_idx = idx2
        else: # sum2 >= mean0 > mean1
            clonal_idx = idx2
            subclone1_idx = idx0
            subclone2_idx = idx1
    else:
        if mean2 > mean1: # mean2 > mean1 >= mean0
            clonal_idx = idx2
            subclone1_idx = idx0
            subclone2_idx = idx1
        else: # mean1 >= mean2, mean1 >= mean0
            clonal_idx = idx1
            subclone1_idx = idx0
            subclone2_idx = idx2         
    
    # Score each cell by the number of mutations 
    partition_vector = np.zeros(M.shape[1])
    partition_vector[subclone1_idx] = 1
    partition_vector[subclone2_idx] = -1
    partition_vector = partition_vector.reshape(-1, 1)
    
    M = M.toarray()
    
    partition_vals = np.matmul(M, partition_vector)
    pred_labels = [(partition_vals[i] >= 0)[0] for i in range(M.shape[0])]
    
    true_labels = [(0 if i in true_part[0] else 1) for i in range(len(true_part[0]) + len(true_part[1]))]
    score = adjusted_rand_score(true_labels, pred_labels)
    
    return score
    


def test_naive_rows_cutpoint(M, true_part, oracle=False):
    """
    Finds a bipartition of the rows using their sums and a cutpoint (all rows with sum > cutpoint are in one set, all rows with sum <= cutpoint are in the other).
    Sort the rows of the matrix by their sum and either:
        Oracle=False: choose the mean of the sums as the cutpoint
        Oracle=True: choose the cutpoint that maximizes ARI with the true_partition
    """

    m, n = M.shape

    # We sort the row sums, and use this order to compute the sets
    rowsums = np.sum(M, axis=1)
    row_idx_pairs = sorted(zip(rowsums, range(len(rowsums))), key=lambda x: x[0], reverse = True)
    row_idx = [x[1] for x in row_idx_pairs]
        
    cell_to_idx = {'cell%d' % c:c for c in range(M.shape[0])}
    true_labels = [(0 if i in true_part[0] else 1) for i in range(len(true_part[0]) + len(true_part[1]))]
            
    if oracle:
        # March down the rows trying to split the rows into 2 sets
        pred_labels = np.zeros(len(row_idx))
        
        best = None
        best_score = -1
        for i in range(m):
            # move cell index i from A to Abar
            pred_labels[row_idx[i]] = 1

            ari = adjusted_rand_score(true_labels, pred_labels)
            if ari > best_score:
                best_score = ari
                best = pred_labels
        score = best_score

    else:
        # Use the mean of the row sums as a threshold
        cutpoint = np.mean(rowsums)
        A = set()
        i = 0
        ordered_rowsums = [int(a[0]) for a in row_idx_pairs]
        while ordered_rowsums[i] > cutpoint:
            A.add('cell%d' % row_idx_pairs[i][1])
            i += 1
        Abar = set()
        for j in range(i, M.shape[0]):
            Abar.add('cell%d' % row_idx_pairs[j][1])
            
        pred_labels = sets_to_clusters((A, Abar, set()), reverse_dict(cell_to_idx))
        score = adjusted_rand_score(true_labels, pred_labels)
    
    return score
    

def kmeans_remove_clonal(M):
    """
    Use k-means to remove mutations that appear to be "clonal".
    Clusters columns by their sum (each column is a point in 1D), and removes all columns in the cluster with the largest mean.
    """

     # Cluster column sums using kmeans
    colsums = np.sum(M, axis=0)
    col_labels = KMeans(n_clusters = 3, random_state = 0).fit_predict(colsums.reshape(-1, 1))
    
    # Call the section with the largest average column sum clonal
    idx0 = []
    idx1 = []
    idx2 = []
    for idx in range(len(col_labels)):
        label = col_labels[idx]
        if label == 0:
            idx0.append(idx)
        elif label == 1:
            idx1.append(idx)
        elif label == 2:
            idx2.append(idx)
        else:
            print("uhoh")

    colsums = colsums.reshape(-1, 1)
    mean0 = np.mean(colsums[idx0])
    mean1 = np.mean(colsums[idx1])
    mean2 = np.mean(colsums[idx2])

    if mean0 > mean1:
        if mean0 > mean2:
            clonal_idx = idx0
            subclone1_idx = idx1
            subclone2_idx = idx2
        else: # sum2 >= mean0 > mean1
            clonal_idx = idx2
            subclone1_idx = idx0
            subclone2_idx = idx1
    else:
        if mean2 > mean1: # mean2 > mean1 >= mean0
            clonal_idx = idx2
            subclone1_idx = idx0
            subclone2_idx = idx1
        else: # mean1 >= mean2, mean1 >= mean0
            clonal_idx = idx1
            subclone1_idx = idx0
            subclone2_idx = idx2         

    return M[:, sorted(list(subclone1_idx) + list(subclone2_idx))]


########## Functions to test regularized spectral clustering proposed by Zhou & Amini (2019) ##########
def adjacency_regularization(A, tau = 3, norm = 'L1'):
    # This function implements Algorithm 1 from Zhou and Amini Feb. 2019 paper in JMLR 
        
    ### Notation correspondence to Algorithm 1 pseudocode ###
    # rdegrees, cdegrees - D_1...D_n
    # rdegrees_sorted, cdegrees_sorted - D_(1) ... D_(n_1)
    # Dbar1/Dbar2 - \overbar{D}
    # alpha1/alpha2 - \alpha
    # threshold1/threshold2 - \hat{d}_1, \hat{d}_2
    # idx1/idx2 - \hat{\mathcal{I}}_1, \hat{\mathcal{I}}_2
    
    assert tau > 0
    assert norm == 'L1' or norm == 'L2'
    
    m, n = A.shape
    
    rdegrees = np.array(np.sum(A, axis=1)).flatten()
    rdegrees_sorted = np.sort(rdegrees)[::-1]
    Dbar1 = np.mean(rdegrees)
    alpha1 = min(int(np.floor(float(m) / Dbar1)), len(rdegrees) - 1)
    threshold1 = tau * rdegrees_sorted[alpha1]
    idx1 = [i for i in range(m) if rdegrees[i] >= threshold1]

    cdegrees = np.array(np.sum(A, axis=0)).flatten()
    cdegrees_sorted = np.sort(cdegrees)[::-1]
    Dbar2 = np.mean(cdegrees)
    alpha2 = min(int(np.floor(float(n) / Dbar2)), len(cdegrees) - 1)
    threshold2 = tau * cdegrees_sorted[alpha2]
    idx2 = [i for i in range(n) if cdegrees[1] >= threshold2]
   
    Are = A.copy().todense()
    # Scale only those elements in rows/columns with degree above thresholds
    if len(idx1) == 0:
        print("No row-wise regularization")
    else:
        if norm == 'L1':
            Are[idx1] = np.diag(float(threshold1) / rdegrees[idx1]) * A[idx1]
        else:
            Are[idx1] = np.diag(np.sqrt(float(threshold1) / rdegrees[idx1])) * A[idx1]
    
    if len(idx2) == 0:
        print("No column-wise regularization")
    else:
        if norm == 'L1':
            Are[:, idx2] = A[:, idx2] * np.diag(float(threshold2) / cdegrees[idx2]) 
        else:
            Are[:, idx2] = A[:, idx2] * np.diag(np.sqrt(float(threshold2) / cdegrees[idx2]))

    return Are

def SC_RRE(A, tau = 1.5, norm = 'L2', n_vectors = None, rows_k = 2, cols_k = 3, classify_cols = False):
    # This function implements Algorithm 4 from Zhou and Amini Feb. 2019 paper in JMLR 
    # Variables correspond with notation in the paper where possible
    m, n = A.shape

    if n_vectors is None:
        n_vectors = min(rows_k, cols_k)
    
    Are = adjacency_regularization(A, tau = tau, norm = norm)
    
    Are = scipy.sparse.lil_matrix(Are)
    Z1, s, Z2T = scipy.sparse.linalg.svds(Are, k=n_vectors)
    s = np.diag(s)
    Z2 = Z2T.transpose()
    
    row_vals = np.matmul(Z1, s)
    col_vals = np.matmul(Z2, s)
    
    dim2 = n_vectors
    
    # Cluster and partition rows
    Z1 = np.matmul(Z1, s).reshape(-1, dim2)
    print(Z1.shape, s.shape, A.shape)
    x_model = KMeans(n_clusters = rows_k, random_state = 0).fit(Z1)
    x_clusters = x_model.predict(Z1)
    
    x_sets = {i:set() for i in range(rows_k)}
    [x_sets[x_clusters[i]].add(i) for i in range(m)]

    if not classify_cols:
        return x_sets

    else:
        # Cluster and partition columns
        Z2 = np.matmul(Z2, s).reshape(-1, dim2)
        y_model = KMeans(n_clusters = cols_k, random_state = 0).fit(Z2)
        y_clusters = y_model.predict(Z2)   

        y_sets = {i:set() for i in range(cols_k)}
        [y_sets[y_clusters[i]].add(i) for i in range(n)]

        return x_sets, y_sets

def test_SCRRE(M, true_labels, n_vectors = 1, tau = 1.5, norm = 'L1'):
    m, _ = M.shape
    x_sets = SC_RRE(M, n_vectors = n_vectors, tau = tau, norm = norm)
    
    assert type(x_sets) == dict
    
    pred_labels = [0 if i in x_sets[0] else 1 for i in range(m)]
    return adjusted_rand_score(true_labels, pred_labels)
    #adjusted_mutual_info_score(true_labels, pred_labels, average_method='arithmetic'), normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')



########## Functions to test spectral clustering ##########
def generate_and_test_firstsplit(design, row_thresholds, col_thresholds, true_first_split = [], cov = 0.01, seed = 0, max_cluster_size = 2, balanced = False):
    mblocks = len(row_thresholds)
    nblocks = len(col_thresholds)
    if len(true_first_split) > 0:
        assert len(true_first_split) == 2
        assert sum([len(a) for a in true_first_split]) == len(row_thresholds)
        assert all(a < mblocks for b in true_first_split for a in b)
        assert all(a >= 0 for b in true_first_split for a in b)
    else:
        true_first_split = [0,1], list(range(2, 8))
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Constructing matrix")
    M0, true_csets, true_ssets = generate_toy_design(design, row_thresholds, col_thresholds, coverage = cov, seed = seed)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Merging cells within clones")
    M1 = merge_perfect(M0, true_csets, max_cluster_size = max_cluster_size, seed = seed, balanced = balanced)
    
    my_truth = set().union(*[true_csets[i] for i in true_first_split[0]]), set().union(*[true_csets[i] for i in true_first_split[1]])

    print("First split: ")
    first_ari, cm1, _, _ = partition_score(M1, my_truth, cmatrix = True)

    return first_ari, cm1.tolist()


def generate_and_test(design, row_thresholds, col_thresholds, true_first_split = [], cov = 0.01, seed = 0, max_cluster_size = 2, balanced = False):
    mblocks = len(row_thresholds)
    nblocks = len(col_thresholds)
    if len(true_first_split) > 0:
        assert len(true_first_split) == 2
        assert sum([len(a) for a in true_first_split]) == len(row_thresholds)
        assert all(a < mblocks for b in true_first_split for a in b)
        assert all(a >= 0 for b in true_first_split for a in b)
    else:
        true_first_split = [0,1], list(range(2, 8))
    
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Constructing matrix")
    M0, true_csets, true_ssets = generate_toy_design(design, row_thresholds, col_thresholds, coverage = cov, seed = seed)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Merging cells within clones")

    M1 = merge_perfect(M0, true_csets, max_cluster_size = max_cluster_size, seed = seed, balanced = balanced)
    print([len(a) for a in true_csets])
    
    my_truth = set().union(*[true_csets[i] for i in true_first_split[0]]), set().union(*[true_csets[i] for i in true_first_split[1]])

    print("First split: ")
    first_ari, cm1, A, Abar = partition_score(M1, my_truth, cmatrix = True)

    if len(A) < len(Abar):
        Mleft = M0[sorted([int(a[4:]) for a in A])]
        leftset = set([int(a[4:]) for a in A])
        Mright = M0[sorted([int(a[4:]) for a in Abar])]
        rightset = set([int(a[4:]) for a in Abar])
    else:
        Mright = M0[sorted([int(a[4:]) for a in A])]
        rightset = set([int(a[4:]) for a in A])
        Mleft = M0[sorted([int(a[4:]) for a in Abar])]
        leftset = set([int(a[4:]) for a in Abar])
        
    # Define the correct partition on the left side according to the cells that were placed there
    left_truth = {i:set() for i in range(len(true_csets))}
    idx1 = 0
    right_truth = {i:set() for i in range(len(true_csets))}
    idx2 = 0
    missing_cells = 0
    for i in range(M1.shape[0]):
        my_sets = [j for j in range(len(true_csets)) if i in true_csets[j]]
        if len(my_sets) == 1:
            my_set = my_sets[0]
        else:
            my_set = len(right_truth) + 1
            
        if i in leftset:
            left_truth[my_set].add(idx1)
            idx1 += 1
        elif i in rightset:
            right_truth[my_set].add(idx2)
            idx2 += 1
        else:
            missing_cells += 1
    print("Found {} missing cells when splitting into left and right".format(missing_cells))
            
    if Mleft.shape[0] > 2:
        print("Merging and partitioning {} cells in left branch: ".format(len(leftset)))
        try:
            M1left = merge_perfect(Mleft, [a for a in list(left_truth.values()) if len(a) > 0], 
                                           max_cluster_size = max_cluster_size, seed = seed, balanced = balanced)
            left_ari, cmL, _, _= partition_score(M1left, left_truth, cmatrix = True)

        except ValueError:
            print("SCIPY VALUE ERROR: too few non-empty cells in this partition")
            left_ari = 0
            cmL = []
    else:
        print("Too few cells in left branch to partition ({})".format(Mleft.shape[0]))
        left_ari = 0
        mL = []     


    #print Mright.shape, np.sum(Mright)
    if Mright.shape[0] > 2:
        print("Merging and partitioning {} cells in right branch: ".format(len(rightset)))
        try:
            M1right = merge_perfect(Mright, [a for a in list(right_truth.values()) if len(a) > 0], 
                                    max_cluster_size = max_cluster_size, seed = seed, balanced = balanced)
            right_ari, cmR, _, _= partition_score(M1right, right_truth, cmatrix = True)
        except ValueError:
            print("SCIPY VALUE ERROR: too few non-empty cells in this partition")
            right_ari = 0    
            cmR = []

    else:
        print("Too few cells in right branch to partition ({})".format(Mright.shape[0]))
        right_ari = 0
        cmR = []

    return first_ari, left_ari, right_ari, (cm1.tolist(), cmL.tolist(), cmR.tolist())



def declonal_cluster(M, true_part, true_clonal = False, cluster_spectral = False):
    """
    Remove clonal SNVs (using k-means or the ground truth) before clustering (using spectral clustering or kmeans)
    """
    ncells, nsnvs = M.shape
    
    if true_clonal:
        # use the true assignment of clonal SNVs
        clonal_idx = [i for i in true_part[2]]
        subclone1_idx = [i for i in true_part[3]]
        subclone2_idx = [i for i in true_part[4]]
    else:
        # Cluster column sums using kmeans
        colsums = np.sum(M, axis=0)
        col_labels = KMeans(n_clusters = 3, random_state = 0).fit_predict(colsums.reshape(-1, 1))
        
        # Call the section with the largest average column sum clonal
        idx0 = []
        idx1 = []
        idx2 = []
        for idx in range(len(col_labels)):
            label = col_labels[idx]
            if label == 0:
                idx0.append(idx)
            elif label == 1:
                idx1.append(idx)
            elif label == 2:
                idx2.append(idx)
            else:
                print("uhoh")

        colsums = colsums.reshape(-1, 1)
        mean0 = np.mean(colsums[idx0])
        mean1 = np.mean(colsums[idx1])
        mean2 = np.mean(colsums[idx2])

        if mean0 > mean1:
            if mean0 > mean2:
                clonal_idx = idx0
                subclone1_idx = idx1
                subclone2_idx = idx2
            else: # sum2 >= mean0 > mean1
                clonal_idx = idx2
                subclone1_idx = idx0
                subclone2_idx = idx1
        else:
            if mean2 > mean1: # mean2 > mean1 >= mean0
                clonal_idx = idx2
                subclone1_idx = idx0
                subclone2_idx = idx1
            else: # mean1 >= mean2, mean1 >= mean0
                clonal_idx = idx1
                subclone1_idx = idx0
                subclone2_idx = idx2         
                
    # Count the number of subclonal mutations in each cell
    rowsum_vector = np.zeros(nsnvs)
    rowsum_vector[subclone1_idx] = 1
    rowsum_vector[subclone2_idx] = 1
    rowsum_vector = rowsum_vector.reshape(-1, 1)
    

    if cluster_spectral:
        M = M[:, sorted(list(subclone1_idx) + list(subclone2_idx))]
        my_cellpart = [['cell{}'.format(a) for a in S] for S in true_part[:2]]
        my_snvpart = [[], [], []]
        
        idx = 0
        for i in range(nsnvs):
            if i in clonal_idx:
                continue
            elif i in subclone1_idx:
                my_snvpart[1].append(idx)
            else:
                assert i in subclone2_idx
                my_snvpart[2].append(idx)
            idx += 1

        return partition_score(M, my_cellpart + my_snvpart, verbose = False)[0]
    else:
        M = M.toarray()
        
        # Cluster row sums using kmeans
        rowsums = np.matmul(M, rowsum_vector)
        pred_labels = KMeans(n_clusters = 2, random_state = 0).fit_predict(rowsums.reshape(-1, 1))
    true_labels = [(0 if i in true_part[0] else 1) for i in range(len(true_part[0]) + len(true_part[1]))]
    
    score = adjusted_rand_score(true_labels, pred_labels)
    return score
    

########## Spectral clustering ##########
def compute_cell_partition(x, xclust=2, cellnames = None, GMM = False):
    """
    Given low-dimensional representation x of rows/cells, cluster into <xclust> clusters using either KMeans (GMM = False) or GMM (GMM = True).
    """
    assert xclust == 2 or xclust == 3
    if cellnames == None:
        cellnames = {i:i for i in range(len(x))}
    else:
        assert len(cellnames) == len(x)

    dim2 = 1 if len(x.shape) == 1 else x.shape[1]

    # cluster cells 
    x = x.reshape(-1, dim2)
    if GMM:
        x_model = GaussianMixture(n_components = xclust, random_state = 0).fit(x)
    else:
        x_model = KMeans(n_clusters = xclust, random_state = 0).fit(x)

    x_clusters = x_model.predict(x)
    
    A = set()
    Abar = set()
    Ox = set()
    for i in range(len(x_clusters)):
        cl = x_clusters[i]
        tkn = cellnames[i]
        if cl == 0:
            A.add(tkn)
        elif cl == 1:
            Abar.add(tkn)
        else:
            assert xclust == 3
            Ox.add(tkn)
    
    if xclust == 3:
        return A, Abar, Ox
    else:
        assert len(Ox) == 0
        return A, Abar

def partition_score(Matrix, true_sets, cmatrix = False, verbose = True, partition_instead_of_sets = False):
    """
    Partition the rows of the matrix using spectral clustering and compare the inferred partition to the true partition using ARI. 
    """

    # compute singular values
    x, y, zero_rows, zero_cols = spectral_partition(Matrix)
    cell_to_idx = {}

    m, n = Matrix.shape

    idx0 = 0
    idx = 0
    for i in range(m):
        if idx0 < len(zero_rows) and zero_rows[idx0] == i: # this cell has degree 0 in the data 
            idx0 += 1 # increment the index in 0 rows list
        else: # cell has some degree in the data
            cell_to_idx['cell%d' % i] = idx  # map the current space in the new list to this cell
            idx += 1 # point to the next space in the new list

    mut_to_idx = {}
    idx0 = 0
    idx = 0
    for i in range(n):
        if idx0 < len(zero_cols) and zero_cols[idx0] == i: # this mut has degree 0 in the data 
            idx0 += 1 # increment the index in 0 cols list
        else: # cell has some degree in the data
            mut_to_idx['mut%d' % i] = idx # map the current space in the new list to this mut
            idx += 1 # point to the next space in the new list

    # run kmeans on the singular values to get set assignments
    A, Abar, B, Bbar, O = compute_partition(x, y, cellnames = reverse_dict(cell_to_idx), mutnames= reverse_dict(mut_to_idx))

    # compute the true and inferred labels
    if partition_instead_of_sets:
        true_labels = true_sets
    else:
        true_labels = []

    pred_labels = []
    n_missing_pred = 0
    n_missing_gt = 0
    for i in range(m):
        cell_key = 'cell{}'.format(i)

        # compute true label if it was not supplied
        if not partition_instead_of_sets:
            my_true_sets = [j for j in range(len(true_sets)) if cell_key in true_sets[j]]
            if len(my_true_sets) == 0:
                #print "Cell missing from ground truth: {}".format(i)
                true_labels.append(len(true_sets))
                n_missing_gt += 1
            else:
                assert len(my_true_sets) == 1
                true_labels.append(my_true_sets[0])

        # compute predicted label
        if cell_key in A:
            pred_labels.append(0)
        elif cell_key in Abar:
            pred_labels.append(1)
        else:
            #print "oops: couldn't find cell <{}>".format(cell_key)
            pred_labels.append(2)
            n_missing_pred += 1
    
    
    assert n_missing_pred == len(zero_rows)

    ari = adjusted_rand_score(true_labels, pred_labels)
    cmat = confusion_matrix(true_labels, pred_labels)
    if verbose:
        if n_missing_pred > 0:
            print("Missing a total of {} cells from A/Abar (empty rows in the matrix)".format(n_missing_pred))
        if n_missing_gt > 0:
            print("Missing a total of {} cells from the ground truth (probably errors from first split)".format(n_missing_gt))
        print(ari)
        print(cmat)
    if cmatrix:
        return ari, cmat, A, Abar
    else:
        return ari, A, Abar

def spectral_partition(M, n_vectors = 1, verbose = False):
    """
    Compute spectral representations of the rows and columns of M as in 
    Dhillon 2001.
    
    Args:
        M (scipy.sparse.coo_matrix): matrix to partition
        n_vectors (int): number of singular vectors to use
        verbose (bool): print the matrix dimensions and singular vectors
    Returns: 
        x (): spectral representation of rows
        y: spectral representation of columns
    """

    if not isinstance(M, scipy.sparse.csr_matrix):
        raise TypeError("Matrix must be in CSR format.")

    m0, n0 = M.shape
    
    rowsums = np.sum(M, axis=1)
    zero_rows = [i for i in range(m0) if rowsums[i] == 0]
    M = csr_delete(M, zero_rows, axis=0)
    rowsums = np.delete(rowsums, zero_rows, axis=0)
    
    if verbose:
        print(M.shape)
    
    colsums = np.sum(M, axis=0).reshape(-1, 1)
    zero_cols = [i for i in range(n0) if colsums[i] == 0]
    M = csr_delete(M, zero_cols, axis=1)    
    colsums = np.delete(colsums, zero_cols, axis=0)

    m, n = M.shape
    if verbose:
        print("Found %d nonzero rows and %d nonzero columns" % (m, n))
    
    # Compute inverse sqrt diagonal matrices for original and transposed weight matrix
    Dx_entries = []
    for i in range(m):
        Dx_entries.append(1 / np.sqrt(float(rowsums[i])))
    a = list(range(m))
    Dx = scipy.sparse.csr_matrix((Dx_entries, (a, a)), shape=(m, m))

        
    Dy_entries = []
    for i in range(n):
        Dy_entries.append(1 / np.sqrt(float(colsums[i])))
    b = list(range(n))
    Dy = scipy.sparse.csr_matrix((Dy_entries, (b, b)), shape=(n, n))

    # Scale the adjacency matrix
    Mhat = Dx * M
    Mhat = Mhat * Dy
    
    # Compute singular values and get the second-largest left and right singular vectors
    U, s, Vt = scipy.sparse.linalg.svds(Mhat, k=n_vectors + 1)
    if verbose:
        print("Singular values: ", s)
 
    xhat = U[:,0:n_vectors]
    yhat = Vt[0:n_vectors,:]
    yhat = yhat.transpose()
        
    # Scale the singular vectors using the diagonal matrices
    x = Dx * xhat
    y = Dy * yhat
        
    return x, y, zero_rows, zero_cols

def compute_partition(x, y, xclust=2, yclust=3, cellnames = None, mutnames = None, together = False, GMM = False):
    """
    Helper function for spectral partitioning as in Dhillon 2001.

    Args:
        x: low-dimensional representation of part 1 vertices
        y: low-dimensional representaiton of part 2 vertices
        xclust: number of clusters for part 1 vertices (default 2)
        yclust: number of clusters for part 2 vertices (default 3)
        cellnames: labels for part 1 vertices (default None)
        mutnames: labels for part 2 vertices (default None)
        together: if True, cluster all vertices simultaneously (default False)
        GMM: if True, use Gaussian mixture model instead of k-means (default False)
    Returns: 
        tuple with sets of partitioned vertices:
            A/Abar(/Ox) partition the part 1 (cell) vertices
            B/Bbar(/Oy) partition the part 2 (mutation) vertices

    Note: the implemention of this function is not very general and will require adjustment if many different values for xclust and yclust are desired.
    """

    assert xclust == 2 or xclust == 3
    assert yclust == 2 or yclust == 3
    if cellnames == None:
        cellnames = {i:i for i in range(len(x))}
    else:
        assert len(cellnames) == len(x)
    
    if mutnames == None:
        mutnames = {i:"mut%d" % i for i in range(len(y))}
    else:
        assert len(mutnames) == len(y)
    
    dim2 = 1 if len(x.shape) == 1 else x.shape[1]

    if together:
        assert xclust == yclust
        # cluster cells and SNVs jointly
        all_data = np.concatenate((x, y)).reshape(-1, dim2)
        if GMM:
            model = GaussianMixture(n_components = xclust, random_state = 0).fit(all_data)
        else:
            model = KMeans(n_clusters = xclust, random_state = 0).fit(all_data)
        all_clusters = model.predict(all_data)
        x_clusters = all_clusters[:len(x)]
        y_clusters = all_clusters[len(x):]

    else: 
        # cluster cells and SNVs separately
        x = x.reshape(-1, dim2)
        y = y.reshape(-1, dim2)
        if GMM:
            x_model = GaussianMixture(n_components = xclust, random_state = 0).fit(x)
            y_model = GaussianMixture(n_components = yclust, random_state = 0).fit(y)
        else:
            x_model = KMeans(n_clusters = xclust, random_state = 0).fit(x)
            y_model = KMeans(n_clusters = yclust, random_state = 0).fit(y)

        x_clusters = x_model.predict(x)
        y_clusters = y_model.predict(y)   
    

    A = set()
    Abar = set()
    Ox = set()
    for i in range(len(x_clusters)):
        cl = x_clusters[i]
        tkn = cellnames[i]
        if cl == 0:
            A.add(tkn)
        elif cl == 1:
            Abar.add(tkn)
        else:
            assert xclust == 3
            Ox.add(tkn)
            
            
    B = set()
    Bbar = set()
    Oy = set()
    for i in range(len(y_clusters)):
        cl = y_clusters[i]
        tkn = mutnames[i]
        if cl == 0:
            B.add(tkn)
        elif cl == 1:
            Bbar.add(tkn)
        else:
            Oy.add(tkn)
    
    if xclust == 3:
        return A, Abar, Ox, B, Bbar, Oy
    else:
        assert len(Ox) == 0
        return A, Abar, B, Bbar, Oy

########## Exhaustive heuristics related to spectral clustering ##########
def find_best_ARI(x, true_cl):
    """
    Given 1-dimensional representation x of objects (here, rows/cells) and a true labeling <true_cl> of these objects, exhaustively finds the cut-point that minimizes the ARI.
    """
    sx = sorted(x)
    cut = 0
    best_ari = -1
    best_cut = 0
    while cut < len(sx):
        my_cl = [(0 if x[i] <= sx[cut] else 1) for i in range(len(x))]
        my_ari = adjusted_rand_score(my_cl, true_cl)
        if my_ari > best_ari:
            best_cut = cut
            best_ari = my_ari
        cut += 1
    return best_cut, best_ari

def find_best_ncut(G, x, y, cell_to_idx=None, mut_to_idx=None):    
    """
    Given bipartite graph G, 1-dimensional representation x of "left" vertices, 1-dimensional representation y of "right" vertices, exhaustively finds the cut-points in these 1D representations that minimizes the normalized cut.
    """

    i2c = reverse_dict(cell_to_idx) if cell_to_idx != None else {i:'cell%d' % i for i in range(len(x))}
    i2m = reverse_dict(mut_to_idx) if mut_to_idx != None else {i:'mut%d' % i for i in range(len(y))}
    first = lambda t : t[0]
    
    xs = sorted([(x[i], i2c[i]) for i in range(len(x))], key=first)
    ys = sorted([(y[i], i2m[i]) for i in range(len(y))], key=first)
    best_ncut = 100
    best_detailed = None
    best_cutpoints = (0, 0, 0)
    
    A = set()
    Abar = set(i2c.values())
    B = set()
    O = set()
    Bbar = set(i2m.values())
    i = 0
    j = 0

    while i < len(xs) - 1:
        # move the cut-point for x over by 1 (loop through all cells with the same x-value)
        while i < len(xs) - 1 and xs[i][0] == xs[i + 1][0]:
            A.add(xs[i][1])
            Abar.remove(xs[i][1])
            i += 1        
            
        while j < len(ys) - 1:
            print(len(O))
            assert len(O) == 0
            # move the first cut-point for y (adding directly from Bbar to B)
            while j < len(ys) - 1 and ys[j][0] == ys[j + 1][0]:
                B.add(ys[j][1])
                Bbar.remove(ys[j][1])
                j += 1
                 
            # initialize O to be as large as possible
            k = len(ys) - 1
            [O.add(ys[l][1]) for l in range(j + 1, len(ys))]
            while k > j:
                # move the second cut-point from the right side, increasing Bbar and decreasing O
                while k > j + 1 and ys[k][0] == ys[k - 1][0]:
                    Bbar.add(ys[k][1])
                    O.remove(ys[k][1])
                    k -= 1

                val = ncut(G, A, Abar, B, Bbar, O, cell_idx=cell_to_idx, mut_idx=mut_to_idx, detailed=True, use_idx=False)
                if sum(val) < best_ncut:
                    best_ncut = sum(val)
                    best_detailed = val
                    best_cutpoints = (i, j, k)
                    
                Bbar.add(ys[k][1])
                O.remove(ys[k][1])
                k -= 1
            B.add(ys[j][1])
            Bbar.remove(ys[j][1])
            j += 1
        A.add(xs[i][1])
        Abar.remove(xs[i][1])
        i += 1
    
    return best_ncut, best_detailed, best_cutpoints

def ncut(G, A, Abar, B, Bbar, O, use_idx=True, cell_idx=None, mut_idx=None, detailed=False):
    """
    Given bipartite graph G with partition {A, Abar} of cell vertices and partition {B, Bbar, O} of mutations vertices, computes the normalized cut objective function from Section 5 of Zha et al., 2001 (http://ranger.uta.edu/~chqding/papers/BipartiteClustering.pdf).
    """

    num1 = 0
    denom1 = 0
    num2 = 0
    denom2 = 0
    
    alpha = 1.0 / (len(B) + len(Bbar) + len(O))
    
    for edge in G.edges:
        source = edge[0]
        dest = edge[1]
        if use_idx:
            if cell_idx != None:
                if type(source) == str:
                    i = 'cell%d' % cell_idx[source]
                    j = 'mut%d' % mut_idx[dest]
                else:
                    i = 'cell%d' % cell_idx[dest]
                    j = 'mut%d' % mut_idx[source]
            else:
                if source.startswith('cell'):
                    i = source
                    j = dest
                elif source.startswith('mut'):
                    i = dest
                    j = source
                else:
                    print("Source does not start with \"cell\" or \"mut\": ", source)
                    return -1
        else:
            if type(source) == str:
                i = source
                j = dest
            elif type(source) == tuple:
                i = dest
                j = source
            else:
                print("Source is not a string or tuple: ", source)
                return -1
        
        if i in A:  
            if j in Bbar:
                num1 += 1
                num2 += 1
                denom2 += 1
            denom1 += 1
        elif i in Abar:
            if j in B:
                num1 += 1
                num2 += 1
                denom1 += 1
            denom2 += 1
            
        else:
            print("Source is not in A or Abar: ", i)
            return -1
    
    if denom1 == 0:
        term1 = 0
    else:
        term1 = float(num1)/float(denom1)
    
    if denom2 == 0:
        term2 = 0
    else:
        term2 = float(num2)/float(denom2) 
    if detailed:
        return term1, term2, alpha * len(O)
    else: 
        return term1 + term2 + alpha * len(O)

def construct_graph(M):
    """
    Given a matrix M, constructs the bipartite graph with biadjacency matrix M.
    """
    G = nx.Graph()
    n_cells, n_muts = M.shape
    
    if sparse.issparse(M):
        M = M.tocoo()
        for i,j, v in zip(M.row, M.col, M.data):
            assert v == 1
            G.add_edge("cell{}".format(i), "mut{}".format(j))    
    else:
        [[G.add_edge("cell%d" % i, "mut%d" % j) for j in range(n_muts) if M[i][j] == 1] for i in range(n_cells)]
    return G