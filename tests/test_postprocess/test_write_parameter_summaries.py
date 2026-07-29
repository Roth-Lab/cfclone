from cfclone.cli import write_parameter_summaries

from tests.helpers import load_config, test_cli_from_config_dict


def test_write_parameter_summaries():

    config_file = "/home/matteo/projects/cfdna/wfs/src/tmp-cfclone/cfclone/tests/test_postprocess/test_write_parameter_summaries.yaml"

    config = load_config(config_file)

    test_cli_from_config_dict(command=write_parameter_summaries, config_dict=config)
