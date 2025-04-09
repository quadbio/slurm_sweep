import slurm_sweep


def test_package_has_version():
    assert slurm_sweep.__version__ is not None
