import os, sys
import re
import numpy as np
import pandas as pd

# Parses BEDPE file outputted by LUMPY into dataframe with one row for each SV/cell pair.
def extractData(bed_evidence_file):
    with open(bed_evidence_file) as evidence_file:
        line = evidence_file.readline()
        sv_data = []

        while line != "":
            elements = line.strip().split('\t')
            # sv lines have 15 elements. Otherwise is an evidence line, skip.
            if len(elements) < 15:
                line = evidence_file.readline()
                continue
            sv_id = elements[6]

            beg_chr = elements[0]
            end_chr = elements[3]
            beg_chr_re = re.search('chr(.*)', beg_chr)
            end_chr_re = re.search('chr(.*)', end_chr)
            if beg_chr_re:
                beg_chr = beg_chr_re.group(1)
            if end_chr_re:
                end_chr = end_chr_re.group(1)
            beg_chr = beg_chr
            end_chr = end_chr

            left_o = elements[8]
            right_o = elements[9]

            # average interval for beginning breakpoint
            beg_loc = (int(elements[1]) + int(elements[2])) / 2
            end_loc = (int(elements[4]) + int(elements[5])) / 2
            sv_type = re.search('TYPE:(.*)', elements[10]).group(1)
            evidence = elements[11].split(';')
            if len(evidence) > 1:
                num_evidence = int(evidence[0].split(',')[1]) + int(evidence[1].split(',')[1])
            else:
                num_evidence = evidence[0].split(',')[1]

            # Get list of evidence lines for this sv
            evidence_lines = []
            for i in range(int(num_evidence)):
                evidence_lines.append(evidence_file.readline())

            # Process evidence lines
            for evidence in evidence_lines:
                evidence = evidence.strip().split('\t')
                barcode = evidence[0]
                sv_data.append(
                    [sv_id, barcode, sv_type, beg_chr, beg_loc, left_o, 
                     end_chr, end_loc, right_o, num_evidence])
            line = evidence_file.readline()
            
    # put into dataframe
    sv_df = pd.DataFrame(sv_data, columns=[
        'sv', 'cell_barcode', 'sv_type', 'left_chr', 'left_loc', 'left_orient', 
        'right_chr', 'right_loc', 'right_orient', 'num_evidence'])
    sv_df = sv_df.drop_duplicates()
    
    return sv_df
   
    

# Filters SVs by sample quality score (SQ) as reported in svtyper genotyping results.
# Returns filtered SVs in dataframe.
def filterVariants(sv_df, gt_filename, min_quality):
    with open(gt_filename) as gt_file:
        gt_line = gt_file.readline()
        # header lines
        while re.match('#.*', gt_line):
            gt_line = gt_file.readline()
            
        remove_rows = []
        previous_sv_id = sv_df.at[0, 'sv']
        
        # Iterate through svs in sv_df and gt_file in parallel. Filter by quality scores reported in gt_file
        for sv_row in sv_df.itertuples():
            sv_id = sv_row.sv
            if sv_id != previous_sv_id:
                gt_line = gt_file.readline()
                columns = gt_line.split('\t')
                gt_sv = columns[2].split('_')[0]
                while str(gt_sv) != str(sv_id):
                    gt_line = gt_file.readline()
                    columns = gt_line.split('\t')
                    gt_sv = columns[2].split('_')[0]

            previous_sv_id = sv_id
             
            # get quality info for sv
            columns = gt_line.split('\t')
            sample_info = columns[9].split(':')
            quality = sample_info[5]  # SQ field in tumor

            # filter by quality
            if (quality == "."):
                quality_yes = False
            else:
                quality_yes = (float(quality) > min_quality)

            # remove svs that do not meet quality threshold
            if not quality_yes:
                index = sv_row.Index
                remove_rows.append(index)
        
    filtered = sv_df.drop(remove_rows)
    filtered = filtered.reset_index(drop=True)
        
    return filtered
            
    
# Reads SV information and creates matrix input for SBMClone. Writes matrix to file. 
# Returns two dictionaries. The first maps cell barcodes to matrix row numbers. The second maps SV ids to matrix col numbers.
def createMatrix(sv_df, sbm_matrix_filename):    
    svs = pd.unique(sv_df['sv'])
    cell_ids = pd.unique(sv_df['cell_barcode'])
    
    # dictionary of cell barcodes to row numbers
    barcode_row_dict = {}
    for idx, barcode in enumerate(cell_ids):
        barcode_row_dict[barcode] = idx

    # dictionary of sv ids to column numbers
    sv_col_dict = {}
    for idx, sv in enumerate(svs):
        sv_col_dict[sv] = idx
   
    # write matrix information
    output_file = sbm_matrix_filename
    with open(output_file, 'w') as file:      
        for sv_row in sv_df.itertuples():
            row = barcode_row_dict.get(sv_row.cell_barcode)
            col = sv_col_dict.get(sv_row.sv)
            file.write("{0},{1}".format(row, col))
            file.write('\n')
       
    return barcode_row_dict, sv_col_dict

   
            
            
            
            