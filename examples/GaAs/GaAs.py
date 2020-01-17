import warnings

from amset.plot import AmsetPlotter
from amset.run import AmsetRunner

settings = {
    "general": {
        "scattering_type": "auto",
        "doping": [3e13],
        "temperatures": [201, 290, 401, 506, 605, 789, 994],
        "bandgap": 1.33,
    },

    "interpolation": {
        "interpolation_factor": 10,
    },

    "performance": {
        "fd_tol": 0.1,
        "dos_estep": 0.01,
    },

    "material": {
        "deformation_potential": (8.6, 8.6),
        "elastic_constant": 139.7,
        "donor_charge": 1,
        "acceptor_charge": 1,
        "static_dielectric": 12.18,
        "high_frequency_dielectric": 10.32,
        "pop_frequency": 8.16
    },

    "output": {
        "separate_scattering_mobilities": True,
        "log_error_traceback": True,
        "print_log": True
    }

}

warnings.simplefilter("ignore")

runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
amset_data = runner.run()

plotter = AmsetPlotter(amset_data)
plt = plotter.plot_rates()
plt.savefig("GaAs_rates.png", bbox_inches="tight", dpi=400)
