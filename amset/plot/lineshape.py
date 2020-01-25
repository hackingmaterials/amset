import logging
from collections import defaultdict

import numpy as np
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from amset.constants import amset_defaults as defaults
from amset.constants import hartree_to_ev, hbar
from amset.electronic_structure.interpolate import Interpolater, get_angstrom_structure
from amset.log import initialize_amset_logger
from amset.plot import BaseAmsetPlotter, amset_base_style
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.plotter import BSPlotter
from sumo.plotting import pretty_plot, styled_plot

logger = logging.getLogger(__name__)


class LineshapePlotter(BaseAmsetPlotter):
    def __init__(self, data, interpolation_factor=1, print_log=defaults["print_log"]):
        super().__init__(data)
        self.interpolation_factor = interpolation_factor
        self.print_log = print_log

    def _get_interpolater(self, n_idx, t_idx):
        # interpolater expects energies in eV and structure in angstrom
        energies = {s: e * hartree_to_ev for s, e in self.energies.items()}
        structure = get_angstrom_structure(self.structure)
        bs = BandStructure(
            self.ir_kpoints,
            energies,
            structure.lattice,
            self.efermi * hartree_to_ev,
            structure=structure,
        )
        nelect = sum([idx for idx in self.vb_idx.values()])

        props = defaultdict(dict)
        for spin in self.spins:
            # easier to interpolate the log
            props[spin]["rates"] = np.log10(
                np.sum(self.scattering_rates[spin][:, n_idx, t_idx], axis=0)
            )

        if self.print_log:
            initialize_amset_logger(filename="amset_bandstructure_plot.log")

        return Interpolater(
            bs,
            nelect,
            interpolation_factor=self.interpolation_factor,
            soc=self.soc,
            other_properties=props,
        )

    @styled_plot(amset_base_style)
    def get_plot(
        self,
        n_idx,
        t_idx,
        zero_to_efermi=True,
        estep=0.01,
        line_density=100,
        height=6,
        width=6,
        ymin=None,
        ymax=None,
        ylabel="Energy (eV)",
        plt=None,
        aspect=None,
        distance_factor=10,
        style=None,
        no_base_style=False,
        fonts=None,
    ):
        interpolater = self._get_interpolater(n_idx, t_idx)

        bs, prop = interpolater.get_line_mode_band_structure(
            line_density=line_density, return_other_properties=True
        )

        emin, emax = self.fd_cutoffs
        if not ymin:
            ymin = emin * hartree_to_ev
            if zero_to_efermi:
                ymin -= bs.efermi

        if not ymax:
            ymax = emax * hartree_to_ev
            if zero_to_efermi:
                ymax -= bs.efermi

        logger.info("Plotting band structure")
        plt = pretty_plot(width=width, height=height, plt=plt)
        ax = plt.gca()

        if zero_to_efermi:
            bs.bands = {s: b - bs.efermi for s, b in bs.bands.items()}
            bs.efermi = 0

        bs_plotter = BSPlotter(bs)
        plot_data = bs_plotter.bs_plot_data(zero_to_efermi=zero_to_efermi)

        energies = np.linspace(ymin, ymax, int((ymax - ymin) / estep))
        distances = np.array([d for x in plot_data["distances"] for d in x])

        # rates are currently log(rate)
        rates = {}
        for spin, spin_data in prop.items():
            rates[spin] = spin_data["rates"]
            rates[spin][rates[spin] <= 0] = np.min(rates[spin][rates[spin] > 0])
            rates[spin][rates[spin] >= 15] = 15

        interp_distances = np.linspace(
            distances.min(), distances.max(), len(distances) * distance_factor
        )

        mesh_data = np.full((len(interp_distances), len(energies)), 1e-2)
        for spin in self.spins:
            for spin_energies, spin_rates in zip(bs.bands[spin], rates[spin]):
                interp_energies = interp1d(distances, spin_energies)(interp_distances)
                spin_rates = savgol_filter(spin_rates, 71, 3)
                interp_rates = interp1d(distances, spin_rates)(interp_distances)
                linewidths = 10 ** interp_rates * hbar / 2

                for d_idx in range(len(interp_distances)):
                    energy = interp_energies[d_idx]
                    linewidth = linewidths[d_idx]

                    broadening = lorentzian(energies, energy, linewidth)
                    mesh_data[d_idx] = np.maximum(broadening, mesh_data[d_idx])
                    mesh_data[d_idx] = np.maximum(broadening, mesh_data[d_idx])

        ax.pcolormesh(
            interp_distances,
            energies,
            mesh_data.T,
            rasterized=True,
            norm=LogNorm(vmin=mesh_data.min(), vmax=mesh_data.max()),
        )

        _maketicks(ax, bs_plotter, ylabel=ylabel)
        _makeplot(
            ax,
            plot_data,
            bs,
            zero_to_efermi=zero_to_efermi,
            width=width,
            height=height,
            ymin=ymin,
            ymax=ymax,
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
        logging.info("\t{:.4f}: {}".format(dist, label))

    ax.set_xticks(unique_d)
    ax.set_xticklabels(unique_l)
    ax.xaxis.grid(False)
    ax.set_ylabel(ylabel)


def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma ** 2)
