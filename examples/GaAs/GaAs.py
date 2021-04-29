import warnings

from amset.core.run import Runner
from amset.plot.rates import RatesPlotter

warnings.simplefilter("ignore")

settings = {
    # general settings
    "doping": [-3e13],
    "temperatures": [201, 290, 401, 506, 605, 789, 994],
    "bandgap": 1.33,
    # electronic_structure settings
    "interpolation_factor": 50,
    # scattering rate settings
    "deformation_potential": (1.2, 8.6),
    "elastic_constant": 139.7,
    "donor_charge": 1,
    "acceptor_charge": 1,
    "static_dielectric": 12.18,
    "high_frequency_dielectric": 10.32,
    "pop_frequency": 8.16,
    # performance settings
    "mobility_rates_only": True,
    "write_mesh": True,
}


if __name__ == "__main__":
    runner = Runner.from_vasprun("vasprun.xml.gz", settings)
    amset_data = runner.run()

    plotter = RatesPlotter(amset_data)
    plt = plotter.get_plot()
    plt.savefig("GaAs_rates.png", bbox_inches="tight", dpi=400)
