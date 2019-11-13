"""
TODO: Add a note about adding new settings (e.g. need to update settings.rst
  in docs, add a default here, add it to the command-line script, and update
  the example settings.yaml file.
"""
__version__ = "0.1.0"

amset_defaults = {

    "general": {
        "doping": [1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21],
        "temperatures": [300],
        "scattering_type": "auto",
        "scissor": None,
        "bandgap": None,
    },

    "interpolation": {
        "kpoints": 150,
        "interpolation_factor": 5,
        "fine_mesh_de": None,
        "use_imp_minimum_mesh": False,
    },

    "material": {
        "high_frequency_dielectric": None,
        "static_dielectric": None,
        "elastic_constant": None,
        "deformation_potential": None,
        "piezeoelectric_coefficient": None,
        "acceptor_charge": 1,
        "donor_charge": 1,
        "pop_frequency": None,
    },

    "performance": {
        "gauss_width": 0.01,
        "energy_cutoff": 1.5,
        "fd_tol": 0.1,
        "ibte_tol": 1,
        "max_ibte_iter": 1,
        "dos_estep": 0.01,
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

