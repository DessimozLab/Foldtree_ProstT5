import AFDB_tools
import os
import shutil	


basedir = snakemake.input[0].split('/')[:-1]
basedir = ''.join( [i + '/' for i in basedir])


with open(snakemake.input[0]) as infile:
	ids = [ i.strip() for i in infile if len(i.strip())>0 ]
resdf = AFDB_tools.grab_entries(ids, verbose = True)
resdf.to_csv(snakemake.output[0])
fasta_str = res2fasta(resdf)
with open(snakemake.output[1], 'w') as outfile:
	outfile.write(fasta_str)

