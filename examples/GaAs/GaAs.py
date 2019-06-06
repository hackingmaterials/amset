import warnings

from amset.log import initialize_amset_logger
from amset.run import AmsetRunner

settings = {
    "general": {
        # "interpolation_factor": 10,
        "interpolation_factor": 50,
        "scattering_type": "auto",
        "doping": [3e13],
        "temperatures": [201, 290, 300, 401, 506, 605, 789, 994],
        "bandgap": 1.33
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
    "performance": {
        "energy_tol": 0.005,
        "energy_cutoff": 1.5,
        "g_tol": 1e-5,
        "max_g_iter": 1,
        "symprec": 0.01,
        "nworkers": -1,
        "dos_estep": 0.001,
    },
    "output": {
        "log_traceback": True,
        "write_mesh": False
    }
}

warnings.simplefilter("ignore")

initialize_amset_logger(log_traceback=settings["output"]["log_traceback"])
# runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
# runner.run()

for i in [80, 100, 120]:
    settings["general"]["interpolation_factor"] = i
    runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
    runner.run()
