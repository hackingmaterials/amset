__version__ = "0.1.0"

amset_defaults = {

    "general": {
        "scissor": None,
        "user_bandgap": None,
        "interpolation_factor": None,
        "scattering": "auto",
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
        "dos_estep": 0.01,
        "dos_width": 0.05,
        "symprec": 0.01,
        "nworkers": -1
    }
}

