import pytest

import pandas as pd 

from pathlib import Path

import yaml

from cfclone.run import SexType

from cfclone.run.fit import fit


def test_fit_all_models():
    
    config = yaml.safe_load(open('/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_fit_output_laplace/test_fit_output_laplace.yaml', 'r'))
    
    out_dir = Path(config['out_dir'])
    
    out_dir.mkdir(exist_ok=True, parents=True)

    # PREPROC INPUT DATA
    
    chroms = config['chromosomes']
    
    clones = config['clones']
    
    if len(chroms) > 0 and not 'all' in chroms:
        
        df = pd.read_csv(config['ctdna_file'], sep='\t')
        
        chroms = ["chr{}".format(c) for c in config['chromosomes']]
        
        df = df.loc[df['chrom'].isin(chroms)]
        
        df.reset_index(drop=True)
        
        proc_ctdna_file = out_dir.joinpath('ctdna.tsv.gz')
    
        df.to_csv(proc_ctdna_file, sep='\t', compression='gzip')
    
        ctdna_file = proc_ctdna_file
        
    else:
        
        ctdna_file = config['ctdna_file']
        
    if len(clones) > 0 and not 'all' in clones:
    
        df_clone = pd.read_csv(config['clone_cn_file'], sep='\t')
        
        df_clone = df_clone.loc[df_clone['clone'].isin(clones)]
        
        df_clone.reset_index(drop=True)
        
        proc_clone_cn_file = out_dir.joinpath('clone.tsv.gz')
        
        df_clone.to_csv(proc_clone_cn_file, sep='\t', compression='gzip')
    
        clone_cn_file = proc_clone_cn_file
    else:
        
        clone_cn_file = config['clone_cn_file']
   
    
    # RUN CFCLONE 
    
    fit_all(
        in_file=ctdna_file,
        clone_cnv_file=clone_cn_file,
        out_dir=out_dir,
        **config['sampler_config'],
    )
    
    var = 1
    
    assert var == 1
    
    
def fit_all(
    clone_cnv_file: str,
    in_file: str,
    out_dir: str,
    add_normal: bool = True,
    exec_dir: bool = False,
    laplace_exec_dir: bool = False,
    num_bins: int | None = None,
    num_chains: int = 12,
    num_rounds: int = 10,
    num_threads: int = 1,
    pi_normal: float = 10.,
    pi_tumour: float = 0.1,
    only_normal: bool = False,
    seed: int | None = None,
    sex: SexType = SexType.female,
    use_clone: tuple = (),
) -> None:
    """Runs all version of cfClone model."""
    
    for use_outlier in [True, False]:
        
        for use_rdr in [True, False]: 
            
            for use_baf in [True, False]:
                
                if not use_rdr and not use_baf:
                    
                    continue
                
                out_dir_model = Path(out_dir).joinpath(
                    "outlier_{outlier}_rdr_{rdr}_baf_{baf}".format(
                        outlier=use_outlier,
                        rdr=use_rdr,
                        baf=use_baf,
                    )
                )
                
                out_dir_model.mkdir(parents=True, exist_ok=True)
                
                out_file_model = str(out_dir_model.joinpath('fit.h5'))
                    
                exec_dir_model = str(out_dir_model.joinpath('exec_dir')) if exec_dir else None
                    
                laplace_dir_model = str(out_dir_model.joinpath('laplace_exec_dir')) if laplace_exec_dir else None
                
                fit(
                    clone_cnv_file=clone_cnv_file,
                    in_file=in_file,
                    out_file=out_file_model,
                    add_normal=add_normal,
                    exec_dir=exec_dir_model,
                    laplace_exec_dir=laplace_dir_model,
                    num_bins=num_bins,
                    num_chains=num_chains,
                    num_rounds=num_rounds,
                    num_threads=num_threads,
                    pi_normal=pi_normal,
                    pi_tumour=pi_tumour,
                    only_normal=only_normal,
                    outlier=use_outlier,
                    rdr=use_rdr,
                    baf=use_baf,
                    use_clone=use_clone,
                    seed=seed,
                    sex=sex,
                )
                
                


    
if __name__ == "__main__":
    
    test_fit_all_models()