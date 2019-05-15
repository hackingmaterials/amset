__version__ = "0.1.0"

amset_defaults = {
    "material": {
        "scissor": None,
        "user_bandgap": None,
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

    "model": {
        "scatterings": "auto",
    },

    "performance": {
        "energy_tol": 0.0001,
        "energy_cutoff": 1.5,
        "g_tol": 1,
        "dos_npoints": 4000
    }
}
