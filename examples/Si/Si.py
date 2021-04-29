import warnings

from amset.core.run import Runner
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
    "deformation_potential": "deformation.h5",
    "elastic_constant": [
        [144, 53, 53, 0, 0, 0],
        [53, 144, 53, 0, 0, 0],
        [53, 53, 144, 0, 0, 0],
        [0, 0, 0, 75, 0, 0],
        [0, 0, 0, 0, 75, 0],
        [0, 0, 0, 0, 0, 75],
    ],
    "static_dielectric": [[11.7, 0, 0], [0, 11.7, 0], [0, 0, 11.7]],
    "high_frequency_dielectric": [[11.7, 0, 0], [0, 11.7, 0], [0, 0, 11.7]],
    # performance settings
    "write_mesh": True,
}

if __name__ == "__main__":
    runner = Runner.from_vasprun("vasprun.xml.gz", settings)
    amset_data = runner.run()

    plotter = RatesPlotter(amset_data)
    plt = plotter.get_plot()
    plt.savefig("Si_rates.png", bbox_inches="tight", dpi=400)
