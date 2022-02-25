import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import SubplotBase
from matplotlib.axis import Axis
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import BSPlotter
from sumo.plotting import pretty_plot
from sumo.symmetry import Kpath, PymatgenKpath

from amset.constants import defaults, hbar
from amset.interpolation.bandstructure import Interpolator
from amset.interpolation.periodic import PeriodicLinearInterpolator
from amset.log import initialize_amset_logger
from amset.plot import BaseMeshPlotter, amset_base_style, styled_plot

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class LineshapePlotter(BaseMeshPlotter):
    def __init__(
        self,
        data,
        interpolation_factor=5,
        print_log=defaults["print_log"],
        symprec=defaults["symprec"],
    ):
        super().__init__(data)
        self.interpolation_factor = interpolation_factor

        if print_log:
            initialize_amset_logger(filename="lineshape.log")
        self.symprec = symprec

    def _get_interpolater(self, n_idx, t_idx, mode="linear"):
        props = defaultdict(dict)
        for spin in self.spins:
            # calculate total rate
            spin_rates = np.sum(self.scattering_rates[spin][:, n_idx, t_idx], axis=0)

            # easier to interpolate the log
            log_rates = np.log10(spin_rates)

            # # handle rates that close to numerical noise
            log_rates[log_rates > 18] = 15
            log_rates[np.isnan(log_rates)] = 15

            # map to full k-point mesh
            props[spin]["rates"] = log_rates

        if mode == "linear":
            return _LinearBandStructureInterpolator(
                self.kpoints,
                self.ir_to_full_kpoint_mapping,
                self.energies,
                self.structure,
                self.efermi,
                props,
            )
        elif mode == "fourier":
            bs = BandStructure(
                self.ir_kpoints,
                self.energies,
                self.structure.lattice,
                self.efermi,
                structure=self.structure,
            )

            return Interpolator(
                bs,
                self.num_electrons,
                interpolation_factor=self.interpolation_factor,
                soc=self.soc,
                other_properties=props,
            )
        raise ValueError("Unknown interpolation mode; should be 'linear' or 'fourier'.")

    @styled_plot(amset_base_style)
    def get_plot(
        self,
        n_idx,
        t_idx,
        zero_to_efermi=True,
        estep=0.01,
        line_density=100,
        height=3.2,
        width=3.2,
        emin=None,
        emax=None,
        amin=5e-5,
        amax=1e-1,
        ylabel="Energy (eV)",
        plt=None,
        aspect=None,
        kpath=None,
        cmap="viridis",
        colorbar=True,
        style=None,
        no_base_style=False,
        fonts=None,
    ):
        interpolater = self._get_interpolater(n_idx, t_idx)

        bs, prop = interpolater.get_line_mode_band_structure(
            line_density=line_density,
            return_other_properties=True,
            kpath=kpath,
            symprec=self.symprec,
        )
        bs, rates = force_branches(bs, {s: p["rates"] for s, p in prop.items()})

        fd_emin, fd_emax = self.fd_cutoffs
        if not emin:
            emin = fd_emin
            if zero_to_efermi:
                emin -= bs.efermi

        if not emax:
            emax = fd_emax
            if zero_to_efermi:
                emax -= bs.efermi

        logger.info("Plotting band structure")
        if isinstance(plt, (Axis, SubplotBase)):
            ax = plt
        else:
            plt = pretty_plot(width=width, height=height, plt=plt)
            ax = plt.gca()

        if zero_to_efermi:
            bs.bands = {s: b - bs.efermi for s, b in bs.bands.items()}
            bs.efermi = 0

        bs_plotter = BSPlotter(bs)
        plot_data = bs_plotter.bs_plot_data(zero_to_efermi=zero_to_efermi)

        energies = np.linspace(emin, emax, int((emax - emin) / estep))
        distances = np.array([d for x in plot_data["distances"] for d in x])

        # rates are currently log(rate)
        mesh_data = np.full((len(distances), len(energies)), 0.0)
        for spin in self.spins:
            for spin_energies, spin_rates in zip(bs.bands[spin], rates[spin]):
                for d_idx in range(len(distances)):
                    energy = spin_energies[d_idx]
                    linewidth = 10 ** spin_rates[d_idx] * hbar / 2
                    broadening = lorentzian(energies, energy, linewidth)
                    broadening /= 1000  # convert 1/eV to 1/meV
                    mesh_data[d_idx] += broadening

        im = ax.pcolormesh(
            distances,
            energies,
            mesh_data.T,
            rasterized=True,
            cmap=cmap,
            norm=LogNorm(vmin=amin, vmax=amax),
            shading="auto",
        )
        if colorbar:
            pos = ax.get_position()
            cax = plt.gcf().add_axes([pos.x1 + 0.035, pos.y0, 0.035, pos.height])
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(axis="y", length=rcParams["ytick.major.size"] * 0.5)
            cbar.ax.set_ylabel(
                r"$A_\mathbf{k}$ (meV$^{-1}$)", rotation=270, va="bottom"
            )

        _maketicks(ax, bs_plotter, ylabel=ylabel)
        _makeplot(
            ax,
            plot_data,
            bs,
            zero_to_efermi=zero_to_efermi,
            width=width,
            height=height,
            ymin=emin,
            ymax=emax,
            aspect=aspect,
        )
        return plt


def _makeplot(
    ax,
    data,
    bs,
    zero_to_efermi=True,
    ymin=-3.0,
    ymax=3.0,
    height=None,
    width=None,
    aspect=None,
):
    """Tidy the band structure & add the density of states if required."""
    # draw line at Fermi level if not zeroing to e-Fermi
    if not zero_to_efermi:
        ytick_color = rcParams["ytick.color"]
        ef = bs.efermi
        ax.axhline(ef, color=ytick_color)

    # set x and y limits
    ax.set_xlim(0, data["distances"][-1][-1])
    if bs.is_metal() and not zero_to_efermi:
        ax.set_ylim(bs.efermi + ymin, bs.efermi + ymax)
    else:
        ax.set_ylim(ymin, ymax)

    # keep correct aspect ratio for axes based on canvas size
    if aspect is not False:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if width is None:
            width = rcParams["figure.figsize"][0]
        if height is None:
            height = rcParams["figure.figsize"][1]

        if not aspect:
            aspect = height / width

        ax.set_aspect(aspect * ((x1 - x0) / (y1 - y0)))


def _maketicks(ax, bs_plotter, ylabel="Energy (eV)"):
    """Utility method to add tick marks to a band structure."""
    # set y-ticks
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # set x-ticks; only plot the unique tick labels
    ticks = bs_plotter.get_ticks()
    unique_d = []
    unique_l = []
    if ticks["distance"]:
        temp_ticks = list(zip(ticks["distance"], ticks["label"]))
        unique_d.append(temp_ticks[0][0])
        unique_l.append(temp_ticks[0][1])
        for i in range(1, len(temp_ticks)):
            # Append label to sequence if it is not same as predecessor
            if unique_l[-1] != temp_ticks[i][1]:
                unique_d.append(temp_ticks[i][0])
                unique_l.append(temp_ticks[i][1])

    logging.info("Label positions:")
    for dist, label in list(zip(unique_d, unique_l)):
        logging.info(f"\t{dist:.4f}: {label}")

    ax.set_xticks(unique_d)
    ax.set_xticklabels(unique_l)
    ax.xaxis.grid(True)
    ax.set_ylabel(ylabel)


def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma**2)


class _LinearBandStructureInterpolator:
    def __init__(
        self,
        full_kpoints,
        ir_to_full_idx,
        energies,
        structure,
        efermi,
        other_properties,
    ):
        self.structure = structure
        self.efermi = efermi
        self.spins = list(energies.keys())
        self.nbands = {s: len(e) for s, e in energies.items()}

        full_energies = {s: e[:, ir_to_full_idx] for s, e in energies.items()}
        self.bs_interpolator = PeriodicLinearInterpolator.from_data(
            full_kpoints, full_energies
        )

        self.property_interpolators = {}
        other_properties = _transpose_dict(other_properties)
        for prop, prop_data in other_properties.items():

            full_prop_data = {s: p[:, ir_to_full_idx] for s, p in prop_data.items()}
            self.property_interpolators[prop] = PeriodicLinearInterpolator.from_data(
                full_kpoints, full_prop_data, gaussian=0.75
            )

    def get_line_mode_band_structure(
        self,
        line_density: int = 50,
        kpath: Optional[Kpath] = None,
        symprec: Optional[float] = defaults["symprec"],
        return_other_properties: bool = False,
    ) -> Union[
        BandStructureSymmLine,
        Tuple[BandStructureSymmLine, Dict[Spin, Dict[str, np.ndarray]]],
    ]:
        """Gets the interpolated band structure along high symmetry directions.

        Args:
            line_density: The maximum number of k-points between each two
                consecutive high-symmetry k-points
            symprec: The symmetry tolerance used to determine the space group
                and high-symmetry path.
            return_other_properties: Whether to include the interpolated
                other_properties data for each k-point along the band structure path.

        Returns:
            The line mode band structure.
        """
        if not kpath:
            kpath = PymatgenKpath(self.structure, symprec=symprec)

        kpoints, labels = kpath.get_kpoints(line_density=line_density, cart_coords=True)
        labels_dict = {
            label: kpoint for kpoint, label in zip(kpoints, labels) if label != ""
        }

        rlat = self.structure.lattice.reciprocal_lattice
        frac_kpoints = rlat.get_fractional_coords(kpoints)

        energies = {}
        other_properties = defaultdict(dict)
        for spin in self.spins:
            energies[spin] = self._interpolate_spin(
                spin, frac_kpoints, self.bs_interpolator
            )

            if return_other_properties:
                for prop, property_interpolator in self.property_interpolators.items():
                    other_properties[spin][prop] = self._interpolate_spin(
                        spin, frac_kpoints, property_interpolator
                    )

        bs = BandStructureSymmLine(
            kpoints,
            energies,
            rlat,
            self.efermi,
            labels_dict,
            coords_are_cartesian=True,
            structure=self.structure,
        )

        if return_other_properties:
            return bs, other_properties
        else:
            return bs

    def _interpolate_spin(self, spin, kpoints, interpolator):
        nkpoints = len(kpoints)
        spin_nbands = self.nbands[spin]
        ibands = np.repeat(np.arange(spin_nbands), nkpoints)
        all_kpoints = np.tile(kpoints, (spin_nbands, 1))
        data = interpolator.interpolate(spin, ibands, all_kpoints)
        return data.reshape(spin_nbands, nkpoints)


def _transpose_dict(d):
    td = defaultdict(dict)
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            td[k2][k1] = v2
    return td


def force_branches(bandstructure, other_property=None):
    """Force a linemode band structure to contain branches.

    Branches give a specific portion of the path from one high-symmetry point
    to another. Branches are required for the plotting methods to function correctly.
    Unfortunately, due to the pymatgen BandStructure implementation they require
    duplicate k-points in the band structure path. To avoid this unnecessary
    computational expense, this function can reconstruct branches in band structures
    without the duplicate k-points.

    Args:
        bandstructure: A band structure object.
        other_property: Another property with the format {spin: (nbands, nkpts, ...)
            to split into branches.

    Returns:
        A band structure with brnaches.
    """
    kpoints = np.array([k.frac_coords for k in bandstructure.kpoints])
    labels_dict = {k: v.frac_coords for k, v in bandstructure.labels_dict.items()}

    # pymatgen band structure objects support branches. These are formed when
    # two kpoints with the same label are next to each other. This bit of code
    # will ensure that the band structure will contain branches, if it doesn't
    # already.
    dup_ids = []
    high_sym_kpoints = tuple(map(tuple, labels_dict.values()))
    for i, k in enumerate(kpoints):
        dup_ids.append(i)
        if (
            tuple(k) in high_sym_kpoints
            and i != 0
            and i != len(kpoints) - 1
            and (
                not np.array_equal(kpoints[i + 1], k)
                or not np.array_equal(kpoints[i - 1], k)
            )
        ):
            dup_ids.append(i)

    kpoints = kpoints[dup_ids]

    eigenvals = {}
    projections = {}
    for spin, spin_energies in bandstructure.bands.items():
        eigenvals[spin] = spin_energies[:, dup_ids]
        if len(bandstructure.projections) != 0:
            projections[spin] = bandstructure.projections[spin][:, dup_ids]

    new_property = {}
    if other_property is not None:
        for spin, spin_prop in other_property.items():
            new_property[spin] = spin_prop[:, dup_ids]

    new_bandstructure = type(bandstructure)(
        kpoints,
        eigenvals,
        bandstructure.lattice_rec,
        bandstructure.efermi,
        labels_dict,
        structure=bandstructure.structure,
        projections=projections,
    )
    return new_bandstructure, new_property
