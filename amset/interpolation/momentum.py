import numpy as np

from amset.interpolation.periodic import (
    PeriodicLinearInterpolator,
    group_bands_and_kpoints,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


class MRTACalculator(PeriodicLinearInterpolator):
    def get_mrta_factor(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        bands, kpoints, single_factor = group_bands_and_kpoints(
            band_a, kpoint_a, band_b, kpoint_b
        )
        p = self.interpolate(spin, bands, kpoints)
        factor = 1 - np.dot(p[0], p[1:].T) / np.linalg.norm(p[0]) ** 2

        if single_factor:
            return factor[0]
        else:
            return factor
