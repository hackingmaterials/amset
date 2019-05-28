import warnings

from amset.log import initialize_amset_logger
from amset.run import AmsetRunner

settings = {
    "general": {
        "user_bandgap": None,
        "interpolation_factor": 3,
        "scattering_type": "auto",
        "doping": [1e15],
        "temperatures": [300]
    },
    "material": {
        "deformation_potential": (6.5, 6.5),
        "elastic_constant": 190,
    },
    "performance": {
        "energy_tol": 0.001,
        "energy_cutoff": 1.5,
        "g_tol": 1,
        "symprec": 0.01,
        "nworkers": -1,

    },
    "output": {
        "log_traceback": True
    }
}

warnings.simplefilter("ignore")

initialize_amset_logger(log_traceback=settings["output"]["log_traceback"])

runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)

runner.run()
