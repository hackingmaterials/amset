from amset.amset import AMSET
import logging
import os

if __name__ == "__main__":
    # user inputs:
    use_single_parabolic_band = False

    logging.basicConfig(level=logging.INFO)

    # setting up the inputs:
    model_params = {"bs_is_isotropic": True, "elastic_scatterings": ["ACD", "IMP", "PIE"],
                    "inelastic_scatterings": ["POP"]}
    if use_single_parabolic_band:
        effective_mass = 0.25
        model_params["poly_bands"]= [[[[0.0, 0.0, 0.0], [0.0, effective_mass]]]]

    performance_params = {"parallel" : True}
    GaAs_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73, "C_el": 139.7,
                   "E_D": {"n": 8.6, "p": 8.6}, "P_PIE": 0.052, "scissor": 0.5818}
    GaAs_path = "../../test_files/GaAs"
    coeff_file = os.path.join(GaAs_path, "fort.123_GaAs_k23")


    AMSET = AMSET(calc_dir=GaAs_path, material_params=GaAs_params,
        model_params = model_params, performance_params= performance_params,
        dopings= [-2e15], temperatures=[300, 600])

    # running AMSET
    AMSET.run(coeff_file=coeff_file)

    # generating files and outputs
    AMSET.write_input_files()
    AMSET.to_csv()
    AMSET.plot(k_plots=['energy'], E_plots='all', show_interactive=True,
               carrier_types=AMSET.all_types, save_format=None)
    AMSET.to_json(kgrid=True, trimmed=True, max_ndata=50, nstart=0)
