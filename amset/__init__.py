__version__ = "0.1.0"

amset_defaults = {

    "general": {
        "scissor": None,
        "bandgap": None,
        "interpolation_factor": 10,
        "scattering_type": "auto",
        "doping": [1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21],
        "temperatures": [300]
    },

    "material": {
        "high_frequency_dielectric": None,
        "static_dielectric": None,
        "elastic_constant": None,
        "deformation_potential_vbm": None,
        "deformation_potential_cbm": None,
        "piezeoelectric_coefficient": None,
        "acceptor_charge": None,
        "donor_charge": None,
        "pop_frequency": None,
        "n_dislocations": None,
        "dislocation_charge": None
    },

    "performance": {
        "energy_tol": 0.001,
        "energy_cutoff": 1.5,
        "g_tol": 1,
        "dos_estep": 0.001,
        "dos_width": 0.01,
        "symprec": 0.01,
        "nworkers": -1,
    },

    "output": {
        "calculate_mobility": True,
        "separate_scattering_mobilities": True,
        "raw_output": False,
        "log_traceback": False
    }
}

