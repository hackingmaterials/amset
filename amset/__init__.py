__version__ = "0.1.0"

amset_defaults = {

    "general": {
        "interpolation_factor": 10,
        "scissor": None,
        "bandgap": None,
        "num_extra_kpoints": None,
        "scattering_type": "auto",
        "doping": [1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21],
        "temperatures": [300]
    },

    "material": {
        "high_frequency_dielectric": None,
        "static_dielectric": None,
        "elastic_constant": None,
        "deformation_potential": None,
        "piezeoelectric_coefficient": None,
        "acceptor_charge": None,
        "donor_charge": None,
        "pop_frequency": None,
    },

    "performance": {
        "gauss_width": 0.001,
        "energy_cutoff": 1.5,
        "fd_tol": 0.005,
        "g_tol": 1,
        "max_g_iter": 1,
        "dos_estep": 0.001,
        "dos_width": None,
        "symprec": 0.01,
        "nworkers": -1,
    },

    "output": {
        "calculate_mobility": True,
        "separate_scattering_mobilities": True,
        "file_format": "json",
        "write_input": True,
        "write_mesh": False,
        "log_error_traceback": False,
        "print_log": False
    }
}

