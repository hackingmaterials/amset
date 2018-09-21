from amset.core import Amset
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

    amset = Amset(calc_dir='.',
                  vasprun_file=os.path.join(PbTe_dir, "vasprun.xml"),
                  material_params=material_params,
                  model_params = model_params,
                  performance_params={"max_nbands": 1, "max_nvalleys": 1},
                  dopings= [-1e19, 1e19],
                  temperatures=[300, 600])

    # running Amset
    amset.run_profiled(coeff_file=coeff_file, kgrid_tp="coarse")

    # generating files and outputs
    amset.write_input_files()
    amset.to_csv()
    amset.plot(k_plots=['energy'], E_plots=['all'],
               show_interactive=True,
               carrier_types=amset.all_types)
    amset.to_file()
    amset.to_json(kgrid=True, trimmed=True, max_ndata=50, n0=0)
