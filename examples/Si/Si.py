import warnings

from amset.plot import AmsetPlotter
from amset.core.run import AmsetRunner

warnings.simplefilter("ignore")

settings = {
    # general settings
    "scattering_type": ["IMP", "ACD"],
    "doping": [1.99e+14, 2.20e+15, 1.72e+16, 1.86e+17, 1.46e+18, 4.39e+18],
    "temperatures": [300],
    "bandgap": 1.14,

    # electronic_structure settings
    "interpolation_factor": 50,

    # materials properties
    "deformation_potential": (6.5, 6.5),
    "elastic_constant": 190,
    "donor_charge": 1,
    "acceptor_charge": 1,
    "static_dielectric": 13.1,
    "high_frequency_dielectric": 13.1,
}

runner = AmsetRunner.from_vasprun("vasprun.xml.gz", settings)
amset_data = runner.run()

plotter = AmsetPlotter(amset_data)
plt = plotter.plot_rates()
plt.savefig("Si_rates.png", bbox_inches="tight", dpi=400)
