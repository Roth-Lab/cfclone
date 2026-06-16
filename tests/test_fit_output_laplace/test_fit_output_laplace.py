import pytest

import pandas as pd 

from pathlib import Path

import yaml

import h5py

import matplotlib.pyplot as plt 

from cfclone.run.postprocess import _load_df

from cfclone.run.fit import fit

from cfclone.run.postprocess import write_samples


def test_fit_output_laplace():
    
    config = yaml.safe_load(open('/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_fit_output_laplace/test_fit_output_laplace.yaml', 'r'))
    
    out_dir = Path(config['out_dir'])
    
    out_dir.mkdir(exist_ok=True, parents=True)

    fit_file = out_dir.joinpath('fit.h5')
    
    if config['pt_report']:
        
        pt_dir = out_dir.joinpath('pt_report')
        
        pt_dir.mkdir(exist_ok=True, parents=True)
        
    else:
        
        pt_dir = None
        
    if config['ls_report']:
    
        lap_dir = out_dir.joinpath('laplace_report')
        
        lap_dir.mkdir(exist_ok=True, parents=True)
    
    else:
        
        lap_dir = None
    
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
    
    fit(
        in_file=ctdna_file,
        clone_cnv_file=clone_cn_file,
        out_file=fit_file,
        # exec_dir=str(pt_dir),
        # laplace_exec_dir=str(lap_dir),
        **config['sampler_config'],
        # seed=seed,
        # num_chains=2,
        # num_rounds=2,
        # num_threads=2,
        # outlier=True,
        # pi_normal=10.,
        # pi_tumour=0.5,
    )
    
    write_samples(in_file=fit_file, out_file=out_dir.joinpath('samples.tsv'))
    
    plot_laplace_trace(in_file=fit_file, out_file=out_dir.joinpath('laplace_trace.pdf'))
    
    var = 1
    
    assert var == 1
    

def _load_results_laplace(file_name):
    
    with h5py.File(file_name) as fh:
        
        samples_df = _load_df(fh, "/results/laplace_samples", downcast=True)

        trace_df = _load_df(fh, "/results/laplace_opt_trace")
        
        trace_df['iteration'] = trace_df['iteration'].apply(pd.to_numeric, downcast='integer')

    return samples_df, trace_df


def plot_laplace_trace(in_file: str, out_file: str) -> None:
    
    samples_laplace_df, trace_laplace_df = _load_results_laplace(in_file)
    
    fig = plt.figure()
    
    gs = fig.add_gridspec(nrows=2, ncols=1)
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    ax1 = fig.add_subplot(gs[1, 0])
    
    ax0.plot(trace_laplace_df['iteration'],trace_laplace_df['log_density'])
    
    ax0.set_ylabel("Log density")
    
    ax0.set_xlabel("Iteration")
    
    ax1.plot(trace_laplace_df['iteration'],trace_laplace_df['step_size'])
    
    ax1.set_ylabel("Step size")
    
    ax1.set_xlabel("Iteration")
    
    fig.align_labels()
    
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=150)
    
    plt.close()


    
if __name__ == "__main__":
    
    test_fit_output_laplace() 