import os
import unittest

from os.path import join as path_join

from amset import amset_defaults
from amset.misc.util import write_settings_to_file, load_settings_from_file

test_dir = os.path.dirname(os.path.realpath(__file__))


class IOTest(unittest.TestCase):

    def test_load_settings_from_file(self):
        """Test loading settings from a file."""

        settings = load_settings_from_file(
            path_join(test_dir, "amset_settings.yaml"))

        # test settings loaded correctly
        self.assertEqual(settings["material"]["scissor"], 3.)
        self.assertEqual(settings["material"]["high_frequency_dielectric"], 10)

        # test defaults inferred correctly
        self.assertEqual(settings["material"]["pop_frequency"],
                         amset_defaults["material"]["pop_frequency"])
        self.assertEqual(settings["performance"]["gauss_width"],
                         amset_defaults["performance"]["gauss_width"])

    def test_write_settings_to_file(self):
        """Test writing settings to a file."""
        settings_file = path_join(path_join(test_dir, "test_settings.yaml"))
        write_settings_to_file(amset_defaults, settings_file)

        with open(settings_file) as f:
            contents = f.read()

        self.assertTrue("material:" in contents)
        self.assertTrue("scatterings: auto" in contents)
        self.assertTrue("    acceptor_charge: null" in contents)

    def tearDown(self):
        settings_file = path_join(path_join(test_dir, "test_settings.yaml"))

        if os.path.exists(settings_file):
            os.remove(settings_file)
