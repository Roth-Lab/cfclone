from cfclone.plot.plot_fit import plot_fit

from tests.helpers import load_config


def test_fit():

    config_file = "/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_plot/test_plot_fit.yaml"

    config = load_config(config_file)

    print("LOL")

    print(config)

    plot_fit(in_file=config["in_file"], out_file=config["out_file"])
