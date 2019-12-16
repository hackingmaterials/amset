import warnings

from amset.plot import AmsetPlotter
from amset.run import AmsetRunner

settings = {
    "general": {
        "scattering_type": ["IMP", "ACD"],
        # "scattering_type": ["ACD"],
        "doping": [1.99e+14, 2.20e+15, 1.72e+16,
                   1.86e+17, 1.46e+18, 4.39e+18],
        "temperatures": [300],
    },

    "interpolation": {
        # "kpoints": None,
        # "interpolation_factor": 600,
        # "fine_mesh_de": 0.02
        # "kpoints": [50, 50, 50],
        "kpoints": None,
        "interpolation_factor": 10,
        # "fine_mesh_de": 0.02
    },

    "material": {
        "deformation_potential": (6.5, 6.5),
        "elastic_constant": 190,
        "donor_charge": 1,
        "acceptor_charge": 1,
        "static_dielectric": 13.1,
        "high_frequency_dielectric": 13.1,
        "pop_frequency": 15.23,
    },

    "performance": {
        "gauss_width": 0.01,
        "fd_tol": 0.15,
        "ibte_tol": 1e-32,
        "max_ibte_iter": 1,
        "dos_estep": 0.01,
        "use_imp_minimum_mesh": False,
    },

    "output": {
        "print_log": True,
        "log_error_traceback": True,
        "write_mesh": True,
    }
}

warnings.simplefilter("ignore")

runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
amset_data = runner.run()

plotter = AmsetPlotter(amset_data)
# plt = plotter.plot_rates(ymin=1e8)
plt = plotter.plot_rates()
plt.savefig("Si_rates.png", bbox_inches="tight", dpi=400)
