import pytest

import pandas as pd 

from pathlib import Path

import yaml

import h5py

import matplotlib.pyplot as plt 

from cfclone.plot.plot_clones import plot_clones, _load_clone_data


def test_plot_clones():
    
    config = yaml.safe_load(open('/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_plot/test_plot_clones.yaml', 'r'))
    
    out_dir = Path(config['out_dir'])
    
    out_dir.mkdir(exist_ok=True, parents=True)

    fit_file = config['fit_file']
    
    plot_clones(fit_file, yvar='total', out_file=out_dir.joinpath('clones_total.png'))
    
    plot_clones(fit_file, yvar='baf', out_file=out_dir.joinpath('clones_baf.png'))
   
    var = 1
    
    assert var == 1
    
if __name__ == "__main__":
    
    test_plot_clones() 