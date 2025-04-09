from unittest.mock import patch

from typer.testing import CliRunner

from slurm_sweep.cli import app

runner = CliRunner()


def test_validate_config(valid_cli_config_file):
    result = runner.invoke(app, ["validate-config", valid_cli_config_file])
    assert result.exit_code == 0
    assert "Configuration file" in result.stdout
    assert "is valid" in result.stdout


def test_configure_sweep(valid_cli_config_file, tmp_path, expected_slurm_script):
    # Define a custom output path for the submission script
    submit_script = tmp_path / "custom_submit.sh"

    with patch("wandb.sweep", return_value="test-sweep-id"):
        # Pass the custom output path to the CLI command
        result = runner.invoke(app, ["configure-sweep", valid_cli_config_file, "--output", str(submit_script)])

        # Verify the CLI command executed successfully
        assert result.exit_code == 0
        assert "Registering the sweep" in result.stdout

        # Verify the generated SLURM script
        assert submit_script.exists()
        with open(submit_script, encoding="utf-8") as f:
            content = f.read()

        # Compare the generated script with the expected script
        assert content == expected_slurm_script
