import os
import sys
from datetime import datetime

def decode_variant(ref, code):
    if code in set('ACGT'):
        return code
    elif code in set('BDHVN.'):
        return None
    else:
        if code == 'R':
            candidates = ['A', 'G']
        elif code == 'Y':
            candidates = ['C', 'T']
        elif code == 'S':
            candidates = ['G', 'C']
        elif code == 'W':
            candidates = ['A', 'T']
        elif code == 'K':
            candidates = ['G', 'T']
        elif code == 'M':
            candidates = ['A', 'C']
        else:
            print(code)
            assert False
        if ref in candidates:
            candidates.remove(ref)
            assert len(candidates) == 1
            return candidates[0]
        else:
            return None

def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    print(datetime.now(), "Starting")
    n_written = 0
    n_multimutlines = 0
    n_skipped = 0


    with open(outfile, 'w') as fout:
        fout.write("#CHR\tPOS\tREF\tVAR\n")
        with open(infile, 'r') as fin:
            fin.readline()
            prev_line = ("chr1", "-1")
            for line in fin:
                chr, pos, ref, code = line.split('\t')[:4]
                if len(chr) > 5:
                    continue
                if (chr, pos) == prev_line:
                    n_multimutlines += 1
                    continue
                else:
                    prev_line = (chr, pos)

                var_allele = decode_variant(ref, code)
                if var_allele is None:
                    n_skipped += 1
                    continue
                else:
                    n_written += 1
                    fout.write("\t".join([chr, pos, ref, var_allele]) + '\n')
    print(datetime.now(), "Done. Wrote {} variants, skipped {} variants that were not the first at the site, skipped {} variants with multiple/ambiguous SNVs.".format(n_written, n_multimutlines, n_skipped))

if __name__ == '__main__':
    n_args = 2
    if len(sys.argv) != n_args + 1:
        print("Incorrect number of arguments (expecting %d, saw %d)" % (n_args, len(sys.argv) - 1))
        quit(1)
    main()