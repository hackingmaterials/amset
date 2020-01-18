import warnings

from amset.plot import AmsetPlotter
from amset.run import AmsetRunner

warnings.simplefilter("ignore")

settings = {
    # general settings
    "doping": [3e13],
    "temperatures": [201, 290, 401, 506, 605, 789, 994],
    "bandgap": 1.33,

    # interpolation settings
    "interpolation_factor": 50,

    # scattering rate settings
    "deformation_potential": (8.6, 8.6),
    "elastic_constant": 139.7,
    "donor_charge": 1,
    "acceptor_charge": 1,
    "static_dielectric": 12.18,
    "high_frequency_dielectric": 10.32,
    "pop_frequency": 8.16
}


runner = AmsetRunner.from_vasprun("vasprun.xml.gz", settings)
amset_data = runner.run()

plotter = AmsetPlotter(amset_data)
plt = plotter.plot_rates()
plt.savefig("GaAs_rates.png", bbox_inches="tight", dpi=400)
