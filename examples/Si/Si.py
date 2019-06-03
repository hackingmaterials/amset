import warnings

from amset.log import initialize_amset_logger
from amset.run import AmsetRunner

settings = {
    "general": {
        "interpolation_factor": 3,
        "scattering_type": "auto",
        "doping": [1.99e+14, 2.20e+15, 1.72e+16,
                   1.86e+17, 1.46e+18, 4.39e+18],
        "temperatures": [300]
    },
    "material": {
        "deformation_potential": (6.5, 6.5),
        "elastic_constant": 190,
        "donor_charge": 1,
        "acceptor_charge": 1,
        "static_dielectric": 13.1,
        "high_frequency_dielectric": 13.0,
        "pop_frequency": 15.23,
    },
    "performance": {
        "energy_tol": 0.005,
        "energy_cutoff": 1.5,
        "g_tol": 1,
        "max_g_iter": 8,
        "symprec": 0.01,
        "nworkers": 1,
        "dos_estep": 0.01,
        "dos_width": 0.01
    },
    "output": {
        "log_traceback": True,
        "write_mesh": True
    }
}

warnings.simplefilter("ignore")

initialize_amset_logger(log_traceback=settings["output"]["log_traceback"])

runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)

runner.run()
