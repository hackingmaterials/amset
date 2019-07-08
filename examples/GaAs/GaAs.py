import warnings

from amset.run import AmsetRunner

settings = {
    "general": {
        "interpolation_factor": 100,
        "scattering_type": "auto",
        "doping": [3e13],
        "temperatures": [201, 290, 300, 401, 506, 605, 789, 994],
        "bandgap": 1.33,
        "num_extra_kpoints": 100
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
        "log_traceback": True,
        "print_log": True
    }
}

warnings.simplefilter("ignore")

runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
runner.run()

# for i in [100, 150, 200, 300, 400, 500]:
#     settings["general"]["interpolation_factor"] = i
#     runner = AmsetRunner.from_vasprun_and_settings("vasprun.xml.gz", settings)
#     runner.run()
