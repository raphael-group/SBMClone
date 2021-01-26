# SBMClone #

SBMClone is a tool that uses stochastic block model (SBM) inference methods to identify clonal structure (groups of cells that share groups of mutations) in low-coverage single-cell DNA sequencing data.
While SBMClone was originally designed for single-nucleotide variants, it can also be applied to other types of mutations such as structural variants. 

SBMclone is described in detail here (open-access):
[Myers, Zaccaria, and Raphael, *Bioinformatics* 2020](https://doi.org/10.1093/bioinformatics/btaa449)

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

This should take no more than a minute to run, and the output should match the contents of the sample_output folder.

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

The SBMClone script infers the size of the matrix from the input data. **Additional guidance on generating input data to SBMClone can be [found below](#generating-input-to-sbmclone)**.



### Running
The command to run SBMClone on input file `matrix.csv` is simply `python sbmclone.py matrix.csv.` Additional command line options can be included between `sbmclone.py` and `matrix.csv`. The inference method is random, so by default SBMClone uses a random number generator seed of 0 to ensure that results are reproducible. This seed can be modified using the `-s` or `--seed` options.

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
The repository also contains several utilities for simulating mutation matrices with various sizes and sets of parameters (in `src/simulation.py`), as well as benchmarking tools including implementations of other methods for partitioning cells such as spectral clustering and naive k-means approaches (in `src/benchmarking.py`). For assistance using these functions, please contact me ([details below](#additional-information)).

----------------

# Generating input to SBMClone #
In this repository we have also included guidance and utility scripts for generating the input data to SBMClone from the following types of data:
* [Single-nucleotide mutations](#single-nucleotide-mutations)
* [Structural variants](#structural-variants)

## Single-nucleotide mutations
### Recommended workflow
The recommended workflow uses several utilities from [CHISEL](https://github.com/raphael-group/chisel/), which can be installed via conda.

1. (skip if all cells are in one file) Merge single-cell BAM files into one file with barcodes using the [chisel_prep command](https://github.com/raphael-group/chisel/blob/master/man/chisel-prep.md)
2. (skip if reads are already aligned) Align reads to preferred reference genome (e.g., using [bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml#command-line)).
3. Call mutations using your favorite variant caller (e.g., [varscan](http://varscan.sourceforge.net/somatic-calling.html)).
4. Convert variant caller output (VCF format) to Mutator input using `varscan2mutator.py`. Example usage to generate `mutator_input.tsv` from `my_variants.vcf`:
```
python3 varscan2mutator.py my_variants.vcf mutator_input.tsv
```
5. Run [CHISEL/Mutator.py](https://github.com/raphael-group/chisel/blob/master/src/chisel/Mutator.py) to extract cell-mutation pairs (see arguments at the top of the script). This script produces a TSV file  with columns CHR (chromosome), POS (position, CELL (cell barcode), MUT (variant allele), MUTCOV (number of reads from this cell containing the variant allele), COV (total number of reads from this cell that cover this position).
6. Construct SBMClone input file [described above](#input) using a subset* of the mutations output by Mutator.

\*In principle, each row in the table could correspond to a 1-entry in the matrix provided to SBMClone. However, it may be useful to restrict the included mutations to those that are present in a sufficient number of cells, are supported by sufficiently many reads, and/or do not appear to be germline mutations (i.e., are absent from the matched normal or pseudonormal sample). For example, in our analysis of the DOP-PCR dataset in the paper, we restricted analysis to mutations with at least 10 total reads, present in <80% of tumor cells, and present in no more than 1 cell in the pseudo-normal sample (see Supplement section S11 for details)

## Structural variants
The repository includes scripts for processing structural variants (SVs) called by LUMPY and producing the corresponding binary mutation matrix, for input into SBMClone. 

### Running LUMPY
In order to use the SV processing scripts, [LUMPY](https://github.com/arq5x/lumpy-sv) must be run with the following specifications:
 
* LUMPY must be run with the `-b` option at runtime, which outputs SVs in BEDPE format instead of VCF. 
* LUMPY must also be run with the `-e` option at runtime, which includes evidence lines for each SV call in the output. This is necessary to retain the single-cell information. 
* The input BAM file for LUMPY should be preprocessed so that for every read, the read name is replaced with the corresponding cellular barcode (@CB field). This is to ensure that when LUMPY is run, the evidence lines for each SV will include the cellular barcode rather than the read name.

The SV processing scripts also include the option for filtering outputted SVs using genotyping stats determined by SVTyper. To use the filtering option, the genotyped VCF file outputted by SVTyper must first be sorted by the ID field. The command to sort the genotyped file `sv.gt.vcf` is:
`grep '^#' sv.gt.vcf > sv.gt.sorted.vcf && grep -v '^#' sv.gt.vcf | sort -nk3 >> sv.gt.sorted.vcf`

### Usage
The command to process a BEDPE file `sv.bedpe`, containing SVs called by LUMPY as specified above, is `python sv_process.py sv.bedpe`. Additional command line options can be included as described below.

### Output
The command line script produces two output files, named with the default prefix "sv_out".
* `sv_out.extracted.csv`: comma-separated text file that contains one line for each SV/cell combination, with information about each SV extracted from the BEDPE file. 
* `sv_out.sbm.csv`: binary mutation matrix in SBMClone format.

### Command line options
```
positional arguments:
  INFILE                SVs called by LUMPY, in BEDPE format

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT             Output prefix (default "sv_out").
  -f SVTYPER_VCF, --filter SVTYPER_VCF
                        Use specified SVTyper output VCF file to filter SVs.
                        The VCF must be sorted by the ID field.
  -q MIN_QUALITY, --quality MIN_QUALITY
                        Filter SVs by minimum sample quality value (SQ)
                        reported by SVTyper.
```

For example, the command to process SVs in `sv.bedpe`, using `sv.gt.sorted.vcf` to filter out SVs with a quality score <100, is `python sv_process.py -f sv.gt.sorted.vcf -q 100 sv.bedpe`.


----------------

# Additional information #
###
For assistance with running SBMClone, interpreting the results, or other related questions, please email me (Matt Myers) at this address: [matt.myers@cs.princeton.edu](mailto:matt.myers@cs.princeton.edu). Many thanks to Claire Du for contributing the structural variant processing scripts and documentation.

### License
See `LICENSE` for license information.

### Citation
Matthew A Myers, Simone Zaccaria, Benjamin J Raphael, Identifying tumor clones in sparse single-cell mutation data, *Bioinformatics*, Volume 36, Issue Supplement_1, July 2020, Pages i186–i193, [https://doi.org/10.1093/bioinformatics/btaa449](https://doi.org/10.1093/bioinformatics/btaa449)
