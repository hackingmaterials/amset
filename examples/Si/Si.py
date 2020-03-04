import warnings

from amset.core.run import AmsetRunner
from amset.plot.rates import RatesPlotter

warnings.simplefilter("ignore")

settings = {
    # general settings
    "scattering_type": ["IMP", "ADP"],
    "doping": [-1.99e14, -2.20e15, -1.72e16, -1.86e17, -1.46e18, -4.39e18],
    "temperatures": [300],
    "bandgap": 1.14,

    # electronic_structure settings
    "interpolation_factor": 50,

    # materials properties
    "deformation_potential": (5.39, 5.39),
    "elastic_constant": 80,
    "donor_charge": 1,
    "acceptor_charge": 1,
    "static_dielectric": 13.1,
    "high_frequency_dielectric": 13.1,
}

runner = AmsetRunner.from_vasprun("vasprun.xml.gz", settings)
amset_data = runner.run()

plotter = RatesPlotter(amset_data)
plt = plotter.get_plot()
plt.savefig("Si_rates.png", bbox_inches="tight", dpi=400)
