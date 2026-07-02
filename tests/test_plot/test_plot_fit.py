from cfclone.plot.plot_fit import plot_fit

from ..helpers import load_config


def test_plot_clones():

    config_file = "/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_plot/test_plot_fit.yaml"

    config, out_dir = load_config(config_file)

    plot_fit(fit_file=config["fit_file"], out_file=out_dir.joinpath("fit.png"))


if __name__ == "__main__":
    test_plot_clones()
