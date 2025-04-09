from unittest.mock import patch

from typer.testing import CliRunner

from slurm_sweep.cli import app

runner = CliRunner()


def test_validate_config(valid_cli_config_file):
    result = runner.invoke(app, ["validate-config", valid_cli_config_file])
    assert result.exit_code == 0
    assert "Configuration file" in result.stdout
    assert "is valid" in result.stdout


def test_configure_sweep(valid_cli_config_file):
    with (
        patch("slurm_sweep.sweep_class.SweepManager.register_sweep") as mock_register,
        patch("slurm_sweep.sweep_class.SweepManager.write_script") as mock_write,
    ):
        mock_register.return_value = None
        mock_write.return_value = None

        result = runner.invoke(app, ["configure-sweep", valid_cli_config_file])

        assert result.exit_code == 0
        assert "Registering the sweep" in result.stdout

        # Verify that the sweep was registered and the script was written
        mock_register.assert_called_once()
        mock_write.assert_called_once()
