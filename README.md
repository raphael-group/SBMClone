# SBMClone #

SBMClone is a tool that uses stochastic block model (SBM) inference methods to identify clonal structure (groups of cells that share groups of mutations) in low-coverage single-cell DNA sequencing data.
While SBMClone was originally designed for single-nucleotide variants, it can also be applied to other types of mutations such as structural variation. 

Setup
------------------------
The setup process for SBMClone requires the following steps:

### Download
The following command clones the current SBMClone repository from GitHub:

    git clone https://github.com/raphael-group/sbmclone.git

### Requirements
The following software is required for SBMClone:

* Python 3
* Python packages: [numpy](https://numpy.org/) [scipy](https://www.scipy.org/scipylib/index.html) [graph-tool](https://graph-tool.skewed.de/) [matplotlib](https://matplotlib.org/)

I recommend using the `conda` environment manager to manage dependencies. With the Python 3 version of [Anaconda](https://www.anaconda.com/products/individual)  or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, the following commands will set up an environment to run SBMClone:
```
conda create -n sbmclone
conda activate sbmclone
conda install numpy scipy matplotlib
conda install -c conda-forge graph-tool 
```

### Testing
With the dependencies set up correctly, the following command will run SBMClone on the provided test input and write the results to a subdirectory called "output":

    python sbmclone.py example-matrix.csv

This should take no more than a minute to run and the output should match the contents of the sample_output folder.

Usage
----------------
### Input
The input to SBMClone is a binary mutation matrix where rows correspond to cells, columns correspond to mutations, and each entry is a 1 if the corresponding cell has the corresponding mutation or 0 otherwise (equivalently, 0-entries could be represented as ?). The input format is a comma-separated text file in which each line encodes the row and column indices of a single 1-entry. For example, a file with the following lines:

```
0,1
3,0
1,1
2,3
3,2
```

specifies a binary mutation matrix with the following form:
```
0 1 0 0
0 1 0 0
1 0 0 1
0 0 1 0
```

The SBMClone script infers the size of the matrix from the input data. 

The inference method is random, so by default SBMClone uses a random-number-generator seed of 0 to ensure that results are reproducible. This seed can be modified using the `-s` or `--seed` options.

### Running
The command to run SBMClone on input file `matrix.csv` is simply `python sbmclone.py matrix.csv.` Additional command line options can be included between `sbmclone.py` and `matrix.csv`.

### Output
SBMClone produces 2 output files in the specified output directory (default directory `output`):
* `cluster-assignments.txt`: comma-separated text file containing block assignments for each row and column in the matrix. The first line contains block labels for rows, and the second contains block assignments for columns. Note that empty rows and columns are not meaningful to the model (as they correspond to cells with no mutations or mutations that are found in no cells), so they are assigned to dummy blocks (last row/column block).
* `blockmatrix.png`: PNG file showing the inferred block structure. Each row in this matrix corresponds to a clone (row block) and each column corresponds to a mutation cluster (column block). Each entry is labeled with the proportion of 1-entries in the corresponding rows and columns, and is shaded likewise. Note that each value is also the maximum-likelihood estimate of the block probability *p* for the corresponding block. This visualization can be disabled with the command-line option `no-visual`.

### SBMClone command line options
```
positional arguments:
  INFILE                                Mutation matrix in CSV format

optional arguments:
  -o OUTDIR                             Output directory (default "output").
  -l MINBLOCKS, --minblocks MINBLOCKS   Minimum number of total blocks to infer (default 4).
  -u MAXBLOCKS, --maxblocks MAXBLOCKS   Maximum number of total blocks to infer (default 1000).
  --hierarchical                        Use the hierarchical SBM for inference (default False)
  --no-visual                           Disable block visualization.
  --no-messages                         Disable the printing of progress and timing information to standard output.
  --seed SEED                           Random seed for replication (default: 0)
```

Programming interface
----------------
The repository also contains several utilities for simulating mutation matrices with various sizes and sets of parameters (in `src/simulation.py`), as well as benchmarking tools including implementations of other methods for partitioning cells such as spectral clustering and naive k-means approaches (in `src/benchmarking.py`). For assistance using these functions, please contact me (details below).

Additional information
----------------
###
For assistance with running SBMClone, interpreting the results, or other related questions, please email me (Matt Myers) at this address: [matt.myers@cs.princeton.edu](mailto:matt.myers@cs.princeton.edu).

### License
See `LICENSE` for license information.

### Citation
(coming soon)
