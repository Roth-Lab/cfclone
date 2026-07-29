from cfclone.cli import write_posterior_predictive

from tests.helpers import load_config, test_cli_from_config_dict


def test_write_posterior_predictive():

    config_file = "/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_postprocess/test_write_posterior_predictive.yaml"

    config = load_config(config_file)

    test_cli_from_config_dict(command=write_posterior_predictive, config_dict=config)
