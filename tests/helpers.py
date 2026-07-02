import pytest
import traceback
from pathlib import Path
import yaml
from click.testing import CliRunner


def load_config(
    config_file: str, make_out_dir: bool = True
) -> dict | tuple[dict, Path] | tuple[dict, Path, Path]:

    config = yaml.safe_load(open(config_file, "r"))

    if config.get("out_dir") is not None:
        out_dir = Path(config["out_dir"])
        if make_out_dir:
            out_dir.mkdir(exist_ok=True, parents=True)

    if config.get("tmp_dir") is not None:
        tmp_dir = Path(config["tmp_dir"])
        if make_out_dir:
            tmp_dir.mkdir(exist_ok=True, parents=True)

    if (config.get("out_dir") is not None) and (config.get("tmp_dir") is not None):
        return config, out_dir, tmp_dir

    elif config.get("out_dir") is not None:
        return config, out_dir

    else:
        return config


def test_cli_from_config_file(command, config_file: str):

    runner = CliRunner()

    result = runner.invoke(command, ["--config-file", config_file])

    print(result.output)

    if result.exit_code != 0:
        print("Command failed with exit code: {}".format(result.exit_code))
        if result.exc_info:
            traceback.print_exception(*result.exc_info)
        if result.exception:
            print("Exception: {}".format(result.exception))
        pytest.fail("exit code == 1")


def test_cli_from_config_dict(command, config_dict: dict):
    runner = CliRunner()

    cli_args = []
    for key, value in config_dict.items():
        cli_args.append("--{}".format(key))
        cli_args.append(str(value))

    result = runner.invoke(command, cli_args)
    print(result.output)

    if result.exit_code != 0:
        print("Command failed with exit code: {}".format(result.exit_code))
        if result.exc_info:
            traceback.print_exception(*result.exc_info)
        if result.exception:
            print("Exception: {}".format(result.exception))
        pytest.fail("exit code == 1")
