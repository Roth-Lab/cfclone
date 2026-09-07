import pandas as pd 

ctdna_file = "example/data/VOA10055P.tsv.gz"
df = pd.read_csv(ctdna_file, sep='\t')
df = df.dropna()
df = df[['chrom', 'start', 'end', 'rdr_cor', 'allele_0_count', 'allele_1_count']]
df = df.rename(columns={'rdr_cor': 'rdr', 'allele_0_count': 'a', 'allele_1_count': 'b'})
# df = df.loc[df['chrom'] == 'chr22']
df.to_csv('example/data/VOA10055P_proc.tsv.gz', sep='\t', index=False, compression='gzip')