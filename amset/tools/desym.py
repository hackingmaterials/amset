# import warnings
# from copy import deepcopy
# from pathlib import Path
#
# import click
# import numpy as np
#
# __author__ = "Alex Ganose"
# __maintainer__ = "Alex Ganose"
# __email__ = "aganose@lbl.gov"
#
# from pymatgen import Spin
#
# warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
#
#
# @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
# @click.argument("sym-folder")
# @click.argument("no-sym-folder")
# @click.option("--symprec", type=float, default=1e-2, help="symmetry precision")
# @click.option("--basic", is_flag=True, default=False, help="use basic mode")
# def desym(sym_folder, no_sym_folder, symprec, basic):
#     """Test desymmetrization routines"""
#     from pymatgen.io.vasp import Vasprun, Wavecar
#
#     from amset.electronic_structure.symmetry import expand_kpoints
#     from amset.log import initialize_amset_logger
#     from amset.wavefunction.common import desymmetrize_coefficients
#     from amset.wavefunction.io import load_coefficients
#
#     initialize_amset_logger(filename=False, print_log=True)
#
#     sym_folder = Path(sym_folder)
#     no_sym_folder = Path(no_sym_folder)
#
#     click.echo("Loading symmetrized wavefunction")
#
#     coeffs_sym, gpoints, kpoints, structure = load_coefficients(sym_folder)
#     full_kpoints, *symmetry_mapping = expand_kpoints(
#         structure, kpoints, time_reversal=True, return_mapping=True, symprec=symprec
#     )
#
#     coeffs_desym = desymmetrize_coefficients(
#         coeffs_sym, gpoints, kpoints, *symmetry_mapping
#     )
#
#     click.echo("Loading unsymmetrized wavefunction")
#     coeffs_no_sym, gpoints_no_sym, no_sym_kpoints, _ = load_coefficients(no_sym_folder)
#     if set(map(tuple, gpoints)) != set(map(tuple, gpoints_no_sym)):
#         raise ValueError("Gpoints do not match")
#
#     # map desym k-points to same order as sym k-points
#     desym_sort_idx = get_sort_idx(full_kpoints)
#     no_sym_sort_idx = get_sort_idx(no_sym_kpoints)
#
#     coeffs_sym_orig = deepcopy(coeffs_sym)
#
#     coeffs_desym_order = {s: c[:, desym_sort_idx] for s, c in coeffs_desym.items()}
#     coeffs_no_sym_order = {s: c[:, no_sym_sort_idx] for s, c in coeffs_no_sym.items()}
#
#     data = check_overlap(coeffs_desym_order, coeffs_no_sym_order)
#     # analyze_bad_overlaps(kpoints, no_sym_kpoints, gpoints, gpoints_no_sym, coeffs_sym_orig, coeffs_no_sym, data, symmetry_mapping, desym_sort_idx)
#     analyze_bad_overlaps(
#         kpoints,
#         full_kpoints,
#         no_sym_kpoints,
#         gpoints,
#         gpoints_no_sym,
#         coeffs_desym,
#         coeffs_no_sym,
#         data,
#         symmetry_mapping,
#         desym_sort_idx,
#     )
#
#
# # @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
# # @click.argument("sym-folder")
# # @click.argument("no-sym-folder")
# # @click.option("--symprec", type=float, default=1e-2, help="symmetry precision")
# # @click.option("--basic", is_flag=True, default=False, help="use basic mode")
# # def desym(sym_folder, no_sym_folder, symprec, basic):
# #     """Test desymmetrization routines"""
# #     from pymatgen.io.vasp import Vasprun, Wavecar
# #
# #     from amset.electronic_structure.symmetry import expand_kpoints
# #     from amset.log import initialize_amset_logger
# #     from amset.wavefunction.common import desymmetrize_coefficients
# #     from amset.wavefunction.vasp import get_wavefunction_coefficients
# #
# #     initialize_amset_logger(filename=False, print_log=True)
# #
# #     sym_folder = Path(sym_folder)
# #     no_sym_folder = Path(no_sym_folder)
# #
# #     click.echo("Loading symmetrized wavefunction")
# #     wave_sym = Wavecar(sym_folder / "WAVECAR")
# #
# #     vr = Vasprun(sym_folder / "vasprun.xml")
# #     structure = vr.final_structure
# #     kpoints = np.array(wave_sym.kpoints)
# #
# #     if basic:
# #         click.echo("Loading unsymmetrized wavefunction")
# #         wave_no_sym = Wavecar(no_sym_folder / "WAVECAR")
# #         desym_basic(structure, wave_sym, wave_no_sym)
# #         import sys
# #
# #         sys.exit()
# #
# #     full_kpoints, *symmetry_mapping = expand_kpoints(
# #         structure, kpoints, time_reversal=True, return_mapping=True, symprec=symprec
# #     )
# #
# #     coeffs_sym, gpoints = get_wavefunction_coefficients(wave_sym, encut=wave_sym.encut)
# #
# #     coeffs_desym = desymmetrize_coefficients(
# #         coeffs_sym, gpoints, kpoints, *symmetry_mapping
# #     )
# #
# #     click.echo("Loading unsymmetrized wavefunction")
# #     # wave_no_sym = Wavecar(no_sym_folder / "WAVECAR")
# #     #
# #     # coeffs_no_sym, gpoints_no_sym = get_wavefunction_coefficients(
# #     #     wave_no_sym, encut=wave_sym.encut
# #     # )
# #     from amset.wavefunction.io import load_coefficients
# #     coeffs_no_sym, gpoints_no_sym, no_sym_kpoints, _ = load_coefficients(no_sym_folder)
# #     if set(map(tuple, gpoints)) != set(map(tuple, gpoints_no_sym)):
# #         raise ValueError("Gpoints do not match")
# #
# #     # map desym k-points to same order as sym k-points
# #     desym_sort_idx = get_sort_idx(full_kpoints)
# #     no_sym_sort_idx = get_sort_idx(no_sym_kpoints)
# #
# #     coeffs_desym = {s: c[:, desym_sort_idx] for s, c in coeffs_desym.items()}
# #     coeffs_no_sym = {s: c[:, no_sym_sort_idx] for s, c in coeffs_no_sym.items()}
# #
# #     data = check_overlap(coeffs_desym, coeffs_no_sym)
# #     analyze_bad_overlaps(wave_sym, None, data, symmetry_mapping, desym_sort_idx)
#
#
# def analyze_bad_overlaps(
#     orig_kpoints,
#     sym_kpoints,
#     nosym_kpoints,
#     sym_gpoints,
#     nosym_gpoints,
#     sym_coeffs,
#     nosym_coeffs,
#     data,
#     symmetry_mapping,
#     sort_idx,
# ):
#     rotations, translations, is_tr, op_mapping, kp_mapping = symmetry_mapping
#     kp_mapping = kp_mapping[sort_idx]
#     op_mapping = op_mapping[sort_idx]
#
#     rots = rotations[op_mapping]
#     tau = translations[op_mapping]
#
#     for spin, spin_data in data.items():
#         if len(spin_data["kpoints"]) == 0:
#             continue
#         bad_kpoints = np.array(spin_data["kpoints"])
#         bad_bands = np.array(spin_data["bands"])
#         diffs = np.array(spin_data["diffs"])
#         # original_k_idxs = kp_mapping[bad_kpoints]
#         # original_k_idxs, unique_idx = np.unique(original_k_idxs, return_index=True)
#         # bad_bands = bad_bands[unique_idx]
#
#         # print(bad_kpoints)
#         # print(len(original_k_idxs), len(bad_kpoints))
#
#         count_actually_bad = 0
#         # for nk, nb, nop, diff in zip(
#         #         original_k_idxs, bad_bands, bad_kpoints[unique_idx], diffs[unique_idx]
#         # ):
#         #     kpoint = orig_kpoints[nk]
#         for nk, nb, nop, diff in zip(bad_kpoints, bad_bands, bad_kpoints, diffs):
#             kpoint = sym_kpoints[sort_idx][nk]
#             original_k = orig_kpoints[kp_mapping][nk]
#             overlap = test_original_overlap(
#                 sym_kpoints,
#                 nosym_kpoints,
#                 sym_gpoints,
#                 nosym_gpoints,
#                 sym_coeffs,
#                 nosym_coeffs,
#                 kpoint,
#                 nb,
#             )
#
#             final_k = np.dot(rots[nop], original_k)
#             kdiff = np.around(final_k)
#             final_k -= kdiff
#
#             edges = np.around(final_k, 5) == -0.5
#             final_k += edges
#             print(original_k.round(4), "->", final_k.round(4))
#             print("  kpoints", kpoint)
#             print("  kdiff", kdiff)
#             print("  edges", edges)
#             print("  rot was", rots[nop].ravel())
#             print("  tau was", tau[nop].ravel())
#             print(" ", overlap)
#
#         print(
#             "{} out of {} are bad due to VASP".format(
#                 count_actually_bad, len(bad_bands)
#             )
#         )
#
#
# # def analyze_bad_overlaps(wave_sym, wave_no_sym, data, symmetry_mapping, sort_idx):
# #     rotations, translations, is_tr, op_mapping, kp_mapping = symmetry_mapping
# #     kp_mapping = kp_mapping[sort_idx]
# #     op_mapping = op_mapping[sort_idx]
# #
# #     rots = rotations[op_mapping]
# #     tau = translations[op_mapping]
# #
# #     for spin, spin_data in data.items():
# #         if len(spin_data["kpoints"]) == 0:
# #             continue
# #         bad_kpoints = np.array(spin_data["kpoints"])
# #         bad_bands = np.array(spin_data["bands"])
# #         diffs = np.array(spin_data["diffs"])
# #         original_k_idxs = kp_mapping[bad_kpoints]
# #         original_k_idxs, unique_idx = np.unique(original_k_idxs, return_index=True)
# #         bad_bands = bad_bands[unique_idx]
# #
# #         print(bad_kpoints)
# #         print(len(original_k_idxs), len(bad_kpoints))
# #
# #         count_actually_bad = 0
# #         for nk, nb, nop, diff in zip(
# #             original_k_idxs, bad_bands, bad_kpoints[unique_idx], diffs[unique_idx]
# #         ):
# #             kpoint = wave_sym.kpoints[nk]
# #             overlap = test_original_overlap(wave_sym, wave_no_sym, kpoint, nb)
# #             final_k = np.dot(rots[nop], kpoint)
# #             final_k -= np.round(final_k)
# #             print(kpoint.round(4), "->", final_k)
# #             if abs(overlap - 1) > 0.005:
# #                 print("  bad in raw data")
# #                 count_actually_bad += 1
# #             else:
# #                 print("  ok in raw data")
# #                 print("  rot was", rots[nop].ravel())
# #                 print("  tau was", tau[nop].ravel())
# #             print(" ", overlap, diff)
# #
# #         print(
# #             "{} out of {} are bad due to VASP".format(
# #                 count_actually_bad, len(bad_bands)
# #             )
# #         )
#
#
# def check_overlap(coeffs_a, coeffs_b):
#     from tqdm.auto import tqdm
#
#     band_idx = 1
#
#     data = {s: {"kpoints": [], "diffs": [], "bands": []} for s in coeffs_a.keys()}
#     for spin, spin_coeffs_a in coeffs_a.items():
#         spin_coeffs_b = coeffs_b[spin]
#
#         if spin_coeffs_a.shape != spin_coeffs_b.shape:
#             raise ValueError("Coefficients shapes do not match")
#
#         diffs = []
#         for nb, nk in tqdm(list(np.ndindex(spin_coeffs_a.shape[:2]))):
#             if nb != band_idx:
#                 continue
#             v_a = spin_coeffs_a[nb, nk]
#             v_b = spin_coeffs_b[nb, nk]
#
#             diff = abs(np.vdot(v_a, v_b))
#             diffs.append(diff)
#
#             if abs(diff - 1) > 0.005:
#                 data[spin]["kpoints"].append(nk)
#                 data[spin]["bands"].append(nb)
#                 data[spin]["diffs"].append(diff)
#
#         diffs = np.array(diffs)
#         bad_overlap = np.abs(diffs - 1) > 0.1
#         n_bad = np.sum(bad_overlap)
#         click.echo("{:.1f} k-points are bad in band {}".format(n_bad, band_idx))
#     return data
#
#
# def get_sort_idx(kpoints):
#     kpoints = np.array(kpoints).round(5)
#     return np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))
#
#
# def desym_basic(structure, wave_sym, wave_no_sym):
#     from amset.electronic_structure.symmetry import expand_kpoints
#
#     (
#         full_kpoints,
#         rotations,
#         translations,
#         is_tri,
#         op_mapping,
#         kp_mapping,
#     ) = expand_kpoints(
#         structure, wave_sym.kpoints, time_reversal=True, return_mapping=True
#     )
#
#     def get_sort_idx_basic(g):
#         g = np.array(g).round(5)
#         g[g == -0.5] = 0.5
#         return np.lexsort((g[:, 2], g[:, 1], g[:, 0]))
#
#     nbmax = wave_sym._nbmax
#     encut = wave_sym.encut
#     wave_const = 0.262465831
#     band_idx = 0
#
#     ops = rotations[op_mapping]
#     taus = translations[op_mapping]
#     tris = is_tri[op_mapping]
#     diffs = []
#     for k_idx in list(range(len(ops))):
#
#         map_idx = kp_mapping[k_idx]
#         op = ops[k_idx]
#         tau = taus[k_idx]
#         kpoint = wave_sym.kpoints[map_idx]
#         gpoints = wave_sym.Gpoints[map_idx]
#         coeffs = wave_sym.coeffs[map_idx]
#
#         rot_kpoint = np.dot(op, kpoint)
#
#         kdiff = np.around(rot_kpoint)
#         rot_kpoint -= kdiff
#
#         edges = np.around(rot_kpoint, 5) == -0.5
#         rot_kpoint += edges
#         kdiff -= edges
#
#         all_g = np.array(list(np.ndindex(tuple(2 * nbmax[::-1] + 1))))
#         all_g = all_g[:, [2, 1, 0]]  # swap columns
#         all_g[all_g[:, 2] > nbmax[2], 2] -= 2 * nbmax[2] + 1
#         all_g[all_g[:, 1] > nbmax[1], 1] -= 2 * nbmax[1] + 1
#         all_g[all_g[:, 0] > nbmax[0], 0] -= 2 * nbmax[0] + 1
#
#         cart_g = np.dot(all_g + rot_kpoint, structure.lattice.reciprocal_lattice.matrix)
#         norm_g = np.linalg.norm(cart_g, axis=1)
#         ener_g = norm_g ** 2 / wave_const
#
#         valid_g = all_g[ener_g <= encut]
#
#         min_g = valid_g.min(axis=0)
#         max_g = valid_g.max(axis=0)
#         ng = max_g - min_g + 1
#
#         shifted_valid_g = valid_g - min_g
#         valid_indices = (
#             shifted_valid_g[:, 0] * ng[1] * ng[2]
#             + shifted_valid_g[:, 1] * ng[2]
#             + shifted_valid_g[:, 2]
#         )
#
#         rot_gpoints = np.dot(op, gpoints.T).T
#         rot_gpoints = np.around(rot_gpoints).astype(int)
#         rot_gpoints += kdiff.astype(int)
#
#         shifted_rot_g = rot_gpoints - min_g
#         rot_indices = (
#             shifted_rot_g[:, 0] * ng[1] * ng[2]
#             + shifted_rot_g[:, 1] * ng[2]
#             + shifted_rot_g[:, 2]
#         )
#
#         if set(rot_indices) != set(valid_indices):
#             print("BAD PLANE WAVE MAPPING")
#
#         f = np.dot(rot_kpoint, tau) + np.dot(rot_gpoints, tau)
#         exp_factor = np.exp(-1j * 2 * np.pi * f)
#         new_coeffs = exp_factor[None, :] * coeffs
#         if tris[k_idx]:
#             new_coeffs = np.conjugate(new_coeffs)
#
#         nosym_kp = get_sort_idx_basic(np.array(wave_no_sym.kpoints))[k_idx]
#
#         nosym_gpoints = wave_no_sym.Gpoints[nosym_kp].astype(int)
#         shifted_nosym_g = nosym_gpoints - min_g
#         nosym_indices = (
#             shifted_nosym_g[:, 0] * ng[1] * ng[2]
#             + shifted_nosym_g[:, 1] * ng[2]
#             + shifted_nosym_g[:, 2]
#         )
#
#         if set(rot_indices) != set(nosym_indices):
#             print("BAD PLANE WAVE MAPPING2")
#
#         v1 = new_coeffs[band_idx][np.argsort(rot_indices)]
#         v2 = wave_no_sym.coeffs[nosym_kp][band_idx][np.argsort(nosym_indices)]
#
#         o1 = abs(np.vdot(v1, v2))
#         o2 = abs(np.vdot(v2, v2))
#         diffs.append(o1 / o2)
#         if abs(o1 / o2 - 1) > 0.05:
#             print(kdiff, edges, np.dot(op, kpoint), taus[k_idx])
#             print(ops[k_idx])
#
#     diffs = np.array(diffs)
#     bad_overlap = np.abs(diffs - 1) > 0.1
#     n_bad = np.sum(bad_overlap)
#     total = n_bad
#     click.echo("{:.1f} k-points are bad".format(total))
#
#
# def test_original_overlap(
#     sym_kpoints,
#     nosym_kpoints,
#     sym_gpoints,
#     nosym_gpoints,
#     sym_coeffs,
#     nosym_coeffs,
#     kpoint,
#     band_idx,
# ):
#     sym_diff = np.linalg.norm(np.array(sym_kpoints) - kpoint, axis=1)
#     if np.min(sym_diff) > 0.01:
#         raise ValueError("k-point is not in symmetrized wavefunction")
#     sym_k_idx = sym_diff.argmin()
#
#     nosym_diff = np.linalg.norm(np.array(nosym_kpoints) - kpoint, axis=1)
#     if np.min(nosym_diff) > 0.01:
#         raise ValueError("k-point is not in no-symmetry wavefunction")
#     nosym_k_idx = nosym_diff.argmin()
#
#     gdiff = np.abs(sym_gpoints - nosym_gpoints)
#     if gdiff.max() > 0.2:
#         ValueError("Gpoints differ!")
#
#     v1 = sym_coeffs[Spin.up][band_idx, sym_k_idx]
#     v1 /= np.linalg.norm(v1)
#
#     v2 = nosym_coeffs[Spin.up][band_idx, nosym_k_idx]
#     v2 /= np.linalg.norm(v2)
#
#     return abs(np.vdot(v1, v2))
#
#
# # def test_original_overlap(wave_sym, wave_nosym, kpoint, band_idx):
# #     sym_diff = np.linalg.norm(np.array(wave_sym.kpoints) - kpoint, axis=1)
# #     if np.min(sym_diff) > 0.01:
# #         raise ValueError("k-point is not in symmetrized wavefunction")
# #     sym_k_idx = sym_diff.argmin()
# #
# #     nosym_diff = np.linalg.norm(np.array(wave_nosym.kpoints) - kpoint, axis=1)
# #     if np.min(nosym_diff) > 0.01:
# #         raise ValueError("k-point is not in no-symmetry wavefunction")
# #     nosym_k_idx = nosym_diff.argmin()
# #
# #     gdiff = np.abs(wave_sym.Gpoints[sym_k_idx] - wave_nosym.Gpoints[nosym_k_idx])
# #     if gdiff.max() > 0.2:
# #         ValueError("Gpoints differ!")
# #
# #     v1 = wave_sym.coeffs[sym_k_idx][band_idx]
# #     v1 /= np.linalg.norm(v1)
# #
# #     v2 = wave_nosym.coeffs[nosym_k_idx][band_idx]
# #     v2 /= np.linalg.norm(v2)
# #
# #     return abs(np.vdot(v1, v2))
