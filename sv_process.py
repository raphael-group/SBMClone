import os, sys 
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # append the root of the repo to the system path
from src.sv_util import * # this line will fail if you move sv_util.py from the parent directory of the src directory

def parse_args():
    description = "Command-line interface for processing SVs called by LUMPY and producing a matrix file in SBMClone format."
    parser = argparse.ArgumentParser(description=description)
    

    parser.add_argument("INFILE", type=str, help="SVs called by LUMPY, in BEDPE format")
    parser.add_argument("-o", metavar="OUTPUT", dest="output", required=False, type=str, 
                        default="sv_out", help="Output prefix (default \"sv_out\").")
    parser.add_argument("-f", "--filter", metavar="SVTYPER_VCF", required=False, type=str, 
                        default=None, help="Use specified SVTyper output VCF file to filter SVs. The VCF must be sorted by the ID field.")
    parser.add_argument("-q", "--quality", metavar="MIN_QUALITY", type=int, dest="quality", required=False,
                        default=None,
                       help="Filter SVs by minimum sample quality value (SQ) reported by SVTyper.")


    args = parser.parse_args()

    if not os.path.isfile(args.INFILE):
        raise ValueError("Input file does not exist: {}".format(args.INFILE))

    if args.quality and not args.filter:
        raise ValueError("To filter SVs, SVTyper VCF file must be specified with the -f option.")
        
    if args.filter and not os.path.isfile(args.filter):
        raise ValueError("Input file does not exist: {}".format(args.filter))
                         
    return {
        "infile" : args.INFILE,
        "output" : args.output,
        "filter": args.filter,
        "quality": args.quality
    }


def run_sv_cl():
    args = parse_args()
    
    sv_df = extractData(args['infile'])
    
    # filter svs
    min_qual = args['quality']
    gt_file = args['filter']
    if gt_file and min_qual:
        sv_df = filterVariants(sv_df, gt_file, min_qual)
     
    # write sv data to file
    prefix = args['output']
    output_sv_file = prefix + '.extracted.csv'
    sv_df.to_csv(output_sv_file, index = False)
    
    # create and write SBMClone matrix to file
    output_mtx_file = prefix + '.sbm.csv'
    createMatrix(sv_df, output_mtx_file)
    
    
if __name__ == '__main__':
    run_sv_cl()

