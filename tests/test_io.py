from pathlib import Path

from amset.constants import defaults
from amset.io import load_settings, write_settings


def test_load_settings(test_dir):
    settings = load_settings(test_dir / "amset_settings.yaml")

    # test settings loaded correctly
    assert settings["scissor"] == 3.0
    assert settings["high_frequency_dielectric"].tolist() == [
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ]

    # test defaults inferred correctly
    assert settings["pop_frequency"] == defaults["pop_frequency"]
    assert settings["dos_estep"] == defaults["dos_estep"]


def test_write_settings(clean_dir):
    settings_file = Path("test_settings.yaml")
    write_settings(defaults, settings_file)
    contents = settings_file.read_text()

    assert "scattering_type: auto" in contents
    assert "defect_charge: 1" in contents
