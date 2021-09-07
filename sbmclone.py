from graph_tool.inference import minimize as gt_min
import os
import sys 
import argparse
import random
import numpy as np 
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # append the root of the repo to the system path
from src.util import * # this line will fail if you move sbmclone.py from the parent directory of the src directory
from datetime import datetime


def parse_args():
    description = "Command-line interface to SBMClone, which infers clones (row blocks) and mutation clusters (column blocks) from a mutation matrix."
    parser = argparse.ArgumentParser(description=description)

    minblocks_default = 4
    maxblocks_default = 1000

    parser.add_argument("INFILE", type=str, help="Mutation matrix in CSV format")
    parser.add_argument("-o", metavar = "OUTDIR", dest = "outdir", required=False, type=str, default="output", help="Output directory (default \"output\").")
    parser.add_argument("-l","--minblocks", type=int, required=False, default=minblocks_default, help="Minimum number of total blocks to infer (default {}).".format(minblocks_default))
    parser.add_argument("-u","--maxblocks", type=int, required=False, default=maxblocks_default, help="Maximum number of total blocks to infer (default {}).".format(maxblocks_default))
    parser.add_argument("--hierarchical", action='store_true', required=False, default=False, help="Use the hierarchical SBM for inference (default False)")
    parser.add_argument("--no-visual", dest = "visualize", action='store_false', required=False, default=True, help="Do not output PNG file with block visualization.")
    parser.add_argument("--no-messages", dest = "verbose", action='store_false', required=False, default=True, help="Disable the printing of progress and timing information to standard output.")
    parser.add_argument("--seed", required=False, type=int, default=None, help="Random seed for replication (default: 0)")

    args = parser.parse_args()

    if not os.path.isfile(args.INFILE):
        raise ValueError("Input file does not exist: {}".format(args.INFILE))
    if not os.path.isdir(args.outdir):
        raise ValueError("Output directory does not exist: {}".format(args.outdir))
    if args.minblocks and args.minblocks <= 0:
        raise ValueError("Minimum number of blocks must be a positive integer (saw {}).".format(args.minblocks))
    if args.maxblocks and args.maxblocks <= 0:
        raise ValueError("Maximum number of blocks must be a positive integer (saw {}).".format(args.maxblocks))
    if args.maxblocks < args.minblocks:
        raise ValueError("Maximum number of blocks must be larger than minimum number of blocks (saw {} < {}, respectively).".format(args.maxblocks, args.minblocks))
    if args.seed and args.seed < 0:
        raise ValueError("The random seed  must be non-negative.")

    return {
        "infile" : args.INFILE,
        "minblocks" : args.minblocks,
        "maxblocks" : args.maxblocks,
        "hierarchical" : args.hierarchical,
        "visualize" : args.visualize,
        "verbose" : args.verbose,
        "outdir" : args.outdir,
        "seed" : args.seed
    }

def logArgs(args, width):
    text = "\n"
    for key in args:
        text += "\t{}: {}\n".format(key, args[key])
    log(msg=text, level="INFO")

def run_sbmclone_cl():
    args = parse_args()
    
    logArgs(args, 80)
    
    verbose = args['verbose']

    seed = args['seed']    
    np.random.seed(seed)
    random.seed(seed)

    if verbose:
        log("Reading input file...")
    M = read_matrix_file(args['infile'], verbose = verbose)

    if verbose:
        log("Inferring block matrix...")
    row_part, col_part = SBMClone(M, min_blocks = args['minblocks'], max_blocks = args['maxblocks'], nested = args['hierarchical'], verbose = verbose)

    if verbose:
        log("Writing block assignments...")
    clusters_outfile = os.path.join(args['outdir'], "cluster-assignments.txt")
    with open(clusters_outfile, "w") as f:
        f.write(",".join([str(a) for a in row_part]) + os.linesep)
        f.write(",".join([str(a) for a in col_part]) + os.linesep)
    
    if args['visualize']:
        if verbose:
            log("Generating block visualization...")
        visualize_matrix(M, row_part, col_part, os.path.join(args['outdir'], "blockmatrix.png"))

    if verbose:
        log("Done.")

def SBMClone(M, min_blocks = 2, max_blocks = None, nested = True, verbose = True):
    """
    Apply SBMClone to the given biadjacency matrix M.
    Returns row and column block assignments.
    """
    assert min_blocks is None or isinstance(min_blocks, int)
    assert max_blocks is None or isinstance(max_blocks, int)
    start = datetime.now()

    m, n = M.shape
    G, label2id, vtype = construct_graph_graphtool(M)
    if nested:
        r = gt_min.minimize_nested_blockmodel_dl(G, B_min = min_blocks, B_max = max_blocks, state_args = {'clabel': vtype})
        b = r.get_bs()[0]
    else:
        r = gt_min.minimize_blockmodel_dl(G, B_min = min_blocks, B_max = max_blocks, state_args = {'clabel': vtype})
        b = r.get_blocks()

    end = datetime.now()
    if verbose:
        log("Block inference elapsed time: {}".format(str(end - start)))

    return blockmodel_to_labels(b, label2id, n_cells = m, n_muts = n)


if __name__ == '__main__':
    run_sbmclone_cl()
