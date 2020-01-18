import os
import unittest
from os.path import join as path_join

from amset.constants import amset_defaults
from amset.util import load_settings_from_file, write_settings_to_file

test_dir = os.path.dirname(os.path.realpath(__file__))


class IOTest(unittest.TestCase):
    def test_load_settings_from_file(self):
        """Test loading settings from a file."""

        settings = load_settings_from_file(path_join(test_dir, "amset_settings.yaml"))

        # test settings loaded correctly
        self.assertEqual(settings["scissor"], 3.0)
        self.assertEqual(settings["high_frequency_dielectric"], 10)

        # test defaults inferred correctly
        self.assertEqual(settings["pop_frequency"], amset_defaults["pop_frequency"])
        self.assertEqual(settings["dos_estep"], amset_defaults["dos_estep"])

    def test_write_settings_to_file(self):
        """Test writing settings to a file."""
        settings_file = path_join(path_join(test_dir, "test_settings.yaml"))
        write_settings_to_file(amset_defaults, settings_file)

        with open(settings_file) as f:
            contents = f.read()

        self.assertTrue("scattering_type: auto" in contents)
        self.assertTrue("acceptor_charge: 1" in contents)

    def tearDown(self):
        settings_file = path_join(path_join(test_dir, "test_settings.yaml"))

        if os.path.exists(settings_file):
            os.remove(settings_file)
