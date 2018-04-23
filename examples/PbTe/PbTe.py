from amset.core import AMSET
import os

if __name__ == "__main__":
    model_params = {"bs_is_isotropic": True,
                    "elastic_scatterings": ["ACD", "IMP", "PIE"],
                    "inelastic_scatterings": ["POP"]}

    material_params = {"epsilon_s": 44.4,
                       "epsilon_inf": 25.6,
                       "W_POP": 10.0,
                       "C_el": 128.8,
                       "E_D": {"n": 4.0, "p": 4.0}}
    PbTe_dir = os.path.join("..", "..", "test_files", "PbTe")
    coeff_file = os.path.join(PbTe_dir, "fort.123")

    AMSET = AMSET(calc_dir='.',
                  vasprun_file=os.path.join(PbTe_dir, "vasprun.xml"),
                  material_params=material_params,
                  model_params = model_params,
                  dopings= [-2e15],
                  temperatures=[300, 600])

    # running AMSET
    AMSET.run(coeff_file=coeff_file, kgrid_tp="very coarse")

    # generating files and outputs
    AMSET.write_input_files()
    AMSET.to_csv()
    AMSET.plot(k_plots=['energy'], E_plots='all', show_interactive=True,
               carrier_types=AMSET.all_types, save_format=None)
    AMSET.to_file()
    AMSET.to_json(kgrid=True, trimmed=True, max_ndata=50, nstart=0)
