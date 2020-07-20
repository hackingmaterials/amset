import logging
from collections import defaultdict

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import SubplotBase
from matplotlib.axis import Axis
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sumo.plotting import pretty_plot, styled_plot

from amset.constants import defaults, hbar
from amset.electronic_structure.interpolate import Interpolater
from amset.log import initialize_amset_logger
from amset.plot import BaseAmsetPlotter, amset_base_style
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.plotter import BSPlotter

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class LineshapePlotter(BaseAmsetPlotter):
    def __init__(self, data, interpolation_factor=5, print_log=defaults["print_log"]):
        super().__init__(data)
        self.interpolation_factor = interpolation_factor

        if print_log:
            initialize_amset_logger(filename="amset_lineshape_plot.log")

    def _get_interpolater(self, n_idx, t_idx):
        # interpolater expects energies in eV and structure in angstrom
        bs = BandStructure(
            self.ir_kpoints,
            self.energies,
            self.structure.lattice,
            self.efermi,
            structure=self.structure,
        )
        nelect = sum([idx for idx in self.vb_idx.values()])

        props = defaultdict(dict)
        for spin in self.spins:
            # easier to interpolate the log
            log_rates = np.log10(
                np.sum(self.scattering_rates[spin][:, n_idx, t_idx], axis=0)
            )
            log_rates[log_rates > 18] = 15
            log_rates[np.isnan(log_rates)] = 15
            props[spin]["rates"] = log_rates

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
        emin=None,
        emax=None,
        ylabel="Energy (eV)",
        plt=None,
        aspect=None,
        distance_factor=10,
        kpath=None,
        style=None,
        no_base_style=False,
        fonts=None,
    ):
        interpolater = self._get_interpolater(n_idx, t_idx)

        bs, prop = interpolater.get_line_mode_band_structure(
            line_density=line_density, return_other_properties=True, kpath=kpath, symprec=None
        )

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
        rates = {}
        for spin, spin_data in prop.items():
            rates[spin] = spin_data["rates"]
            rates[spin][rates[spin] <= 0] = np.min(rates[spin][rates[spin] > 0])
            rates[spin][rates[spin] >= 15] = 15

        interp_distances = np.linspace(
            distances.min(), distances.max(), int(len(distances) * distance_factor)
        )

        window = np.min([len(distances) - 2, 71])
        window += window % 2 + 1
        mesh_data = np.full((len(distances), len(energies)), 1e-2)

        for spin in self.spins:
            for spin_energies, spin_rates in zip(bs.bands[spin], rates[spin]):
                # interp_energies = interp1d(distances, spin_energies)(interp_distances)
                # spin_rates = savgol_filter(spin_rates, window, 3)
                # interp_rates = interp1d(distances, spin_rates)(interp_distances)
                # linewidths = 10 ** interp_rates * hbar / 2

                # for d_idx in range(len(interp_distances)):
                for d_idx in range(len(distances)):
                    # energy = interp_energies[d_idx]
                    energy = spin_energies[d_idx]
                    linewidth = 10 ** spin_rates[d_idx] * hbar / 2
                    # linewidth = linewidths[d_idx]

                    broadening = lorentzian(energies, energy, linewidth)
                    mesh_data[d_idx] = np.maximum(broadening, mesh_data[d_idx])
                    mesh_data[d_idx] = np.maximum(broadening, mesh_data[d_idx])

        ax.pcolormesh(
            distances,
            energies,
            mesh_data.T,
            rasterized=True,
            # cmap="terrain_r",
            # cmap="viridis",
            cmap="viridis",
            norm=LogNorm(vmin=mesh_data.min(), vmax=mesh_data.max()),
            # norm=Normalize(vmin=mesh_data.min(), vmax=mesh_data.max())),
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
        logging.info("\t{:.4f}: {}".format(dist, label))

    ax.set_xticks(unique_d)
    ax.set_xticklabels(unique_l)
    ax.xaxis.grid(True)
    ax.set_ylabel(ylabel)


def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma ** 2)
