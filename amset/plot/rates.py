import abc
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cauchy
from BoltzTraP2 import units
from matplotlib import cycler
from monty.serialization import loadfn

from amset.constants import bohr_to_cm, hartree_to_ev, hbar
from amset.constants import amset_defaults as defaults
from amset.core.data import AmsetData
from amset.electronic_structure.interpolate import Interpolater
from amset.log import initialize_amset_logger
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.plotter import BSPlotter

class AmsetBandStructurePlotter(BaseAmsetPlotter):

    def __init__(
        self,
        data,
        interpolation_factor=defaults["interpolation_factor"],
        print_log=defaults["print_log"],
    ):
        super().__init__(data)

        # interpolater expects energies in eV
        energies = {s: e * hartree_to_ev for s, e in self.energies.items()}
        bs = BandStructure(
            self.ir_kpoints,
            energies,
            self.structure.lattice,
            self.efermi * hartree_to_ev
        )
        nelect = sum([idx for idx in self.vb_idx])

        props = defaultdict(dict)
        for spin in self.spins:
            for n, t in np.ndindex(self.fermi_levels.shape):
                name = "{}-{}".format(n, t)
                props[spin][name] = np.sum(self.scattering_rates[spin][:, n, t], axis=0)

        if print_log:
            initialize_amset_logger(filename="amset_bandstructure_plot.log")

        self.interpolater = Interpolater(
            bs,
            nelect,
            interpolation_factor=interpolation_factor,
            soc=self.soc,
            other_properties=props
        )

    def get_plot(self, n_idx, t_idx, zero_to_efermi=True, estep=0.001):
        bs, prop = self.interpolater.get_line_mode_band_structure(
            return_other_properties=True
        )

        # get the linewidths for the doping and temperatures we want
        name = "{}-{}".format(n_idx, t_idx)
        linewidths = {s: d[name] for s, d in prop.items()}

        bs_plotter = BSPlotter(bs)
        plot_data = bs_plotter.bs_plot_data(zero_to_efermi=zero_to_efermi)

        emin, emax = self.fd_tols
        emin *= hartree_to_ev
        emax *= hartree_to_ev

        energies = np.linspace(emin, emax, int((emax - emin) / estep))
        distances = plot_data["distances"]

        mesh_data = np.zeros((len(distances), len(energies)))
        for d_idx in enumerate(distances):
            for spin in self.spins:
                for b_idx, band_energies in bs.bands[spin].items():
                    lw = linewidths[spin][b_idx, d_idx] * hbar
                    energy = band_energies[d_idx]

                    mesh_data[d_idx] += cauchy.pdf(energies, loc=energy, scale=lw)

        ax = plt.gca()
        ax.pcolormesh(distances, energies, mesh_data.T)

    def _makeplot(self, ax, fig, data, zero_to_efermi=True,
                  vbm_cbm_marker=False, ymin=-6., ymax=6.,
                  height=None, width=None,
                  dos_plotter=None, dos_options=None, dos_label=None,
                  aspect=None):
        """Tidy the band structure & add the density of states if required."""
        # draw line at Fermi level if not zeroing to e-Fermi
        if not zero_to_efermi:
            ytick_color = rcParams['ytick.color']
            ef = self._bs.efermi
            ax.axhline(ef, color=ytick_color)

        # set x and y limits
        ax.set_xlim(0, data['distances'][-1][-1])
        if self._bs.is_metal() and not zero_to_efermi:
            ax.set_ylim(self._bs.efermi + ymin, self._bs.efermi + ymax)
        else:
            ax.set_ylim(ymin, ymax)

        if vbm_cbm_marker:
            for cbm in data['cbm']:
                ax.scatter(cbm[0], cbm[1], color='C2', marker='o', s=100)
            for vbm in data['vbm']:
                ax.scatter(vbm[0], vbm[1], color='C3', marker='o', s=100)

        if dos_plotter:
            ax = fig.axes[1]

            if not dos_options:
                dos_options = {}

            dos_options.update({'xmin': ymin, 'xmax': ymax})
            self._makedos(ax, dos_plotter, dos_options, dos_label=dos_label)
        else:
            # keep correct aspect ratio for axes based on canvas size
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            if width is None:
                width = rcParams['figure.figsize'][0]
            if height is None:
                height = rcParams['figure.figsize'][1]

            if not aspect:
                aspect = height / width

            ax.set_aspect(aspect * ((x1 - x0) / (y1 - y0)))

    def _makedos(self, ax, dos_plotter, dos_options, dos_label=None):
        """This is basically the same as the SDOSPlotter get_plot function."""

        # don't use first 4 colours; these are the band structure line colours
        cycle = cycler(
            'color', rcParams['axes.prop_cycle'].by_key()['color'][4:])
        with context({'axes.prop_cycle': cycle}):
            plot_data = dos_plotter.dos_plot_data(**dos_options)

        mask = plot_data['mask']
        energies = plot_data['energies'][mask]
        lines = plot_data['lines']
        spins = [Spin.up] if len(lines[0][0]['dens']) == 1 else \
            [Spin.up, Spin.down]

        for line_set in plot_data['lines']:
            for line, spin in it.product(line_set, spins):
                if spin == Spin.up:
                    label = line['label']
                    densities = line['dens'][spin][mask]
                else:
                    label = ""
                    densities = -line['dens'][spin][mask]
                ax.fill_betweenx(energies, densities, 0, lw=0,
                                 facecolor=line['colour'],
                                 alpha=line['alpha'])
                ax.plot(densities, energies, label=label,
                        color=line['colour'])

            # x and y axis reversed versus normal dos plotting
            ax.set_ylim(dos_options['xmin'], dos_options['xmax'])
            ax.set_xlim(plot_data['ymin'], plot_data['ymax'])

            if dos_label is not None:
                ax.set_xlabel(dos_label)

        ax.set_xticklabels([])
        ax.legend(loc=2, frameon=False, ncol=1, bbox_to_anchor=(1., 1.))

    @staticmethod
    def _sanitise_label(label):
        """Implement label hacks: Hide trailing @, remove label with leading @
        """

        import re
        if re.match('^@.*$', label):
            return None
        else:
            return re.sub('@+$', '', label)

    @classmethod
    def _sanitise_label_group(cls, labelgroup):
        """Implement label hacks: Hide trailing @, remove label with leading @

        Labels split with $\mid$ symbol will be treated for each part.
        """

        if r'$\mid$' in labelgroup:
            label_components = labelgroup.split(r'$\mid$')
            good_labels = [l for l in
                           map(cls._sanitise_label, label_components)
                           if l is not None]
            if len(good_labels) == 0:
                return None
            else:
                return r'$\mid$'.join(good_labels)
        else:
            return cls._sanitise_label(labelgroup)

    def _maketicks(self, ax, ylabel='Energy (eV)'):
        """Utility method to add tick marks to a band structure."""
        # set y-ticks
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # set x-ticks; only plot the unique tick labels
        ticks = self.get_ticks()
        unique_d = []
        unique_l = []
        if ticks['distance']:
            temp_ticks = list(zip(ticks['distance'], ticks['label']))
            unique_d.append(temp_ticks[0][0])
            unique_l.append(temp_ticks[0][1])
            for i in range(1, len(temp_ticks)):
                # Hide labels marked with @
                if '@' in temp_ticks[i][1]:
                    # If a branch connection, check all parts of label
                    if r'$\mid$' in temp_ticks[i][1]:
                        label_components = temp_ticks[i][1].split(r'$\mid$')
                        good_labels = [l for l in label_components
                                       if l[0] != '@']
                        if len(good_labels) == 0:
                            continue
                        else:
                            temp_ticks[i] = (temp_ticks[i][0],
                                             r'$\mid$'.join(good_labels))
                    # If a single label, check first character
                    elif temp_ticks[i][1][0] == '@':
                        continue

                # Append label to sequence if it is not same as predecessor
                if unique_l[-1] != temp_ticks[i][1]:
                    unique_d.append(temp_ticks[i][0])
                    unique_l.append(temp_ticks[i][1])

        logging.info('Label positions:')
        for dist, label in list(zip(unique_d, unique_l)):
            logging.info('\t{:.4f}: {}'.format(dist, label))

        ax.set_xticks(unique_d)
        ax.set_xticklabels(unique_l)
        ax.xaxis.grid(True, ls='-')
        ax.set_ylabel(ylabel)

        trans_xdata_yaxes = blended_transform_factory(ax.transData,
                                                      ax.transAxes)
        ax.vlines(unique_d, 0, 1,
                  transform=trans_xdata_yaxes,
                  colors=rcParams['grid.color'],
                  linewidth=rcParams['grid.linewidth'],
                  zorder=3)





class AmsetRatesPlotter(BaseAmsetPlotter):
    def plot_rates(
        self,
        plot_fd_tols: bool = True,
        plot_total_rate: bool = False,
        ymin: float = None,
        ymax: float = None,
        normalize_energy: bool = True,
    ):
        if normalize_energy and self.is_metal:
            norm_e = self.fermi_levels[0][0]
        elif normalize_energy:
            cb_idx = {s: v + 1 for s, v in self.vb_idx.items()}
            norm_e = np.max([self.energies[s][: cb_idx[s]] for s in self.spins])
        else:
            norm_e = 0

        norm_e /= units.eV

        energies = np.vstack([self.energies[s] for s in self.spins])
        rates = np.concatenate([self.scattering_rates[s] for s in self.spins], axis=3)

        n_dopings = len(self.doping)
        n_temperatures = len(self.temperatures)
        base_size = 4

        if n_dopings == 1:
            figsize = (1.2 * base_size * n_temperatures, base_size)
            fig, axes = plt.subplots(1, n_temperatures, figsize=figsize)
        elif n_temperatures == 1:
            figsize = (1.2 * base_size * n_dopings, base_size)
            fig, axes = plt.subplots(1, n_dopings, figsize=figsize)
        else:
            figsize = (1.2 * base_size * n_dopings, base_size * n_temperatures)
            fig, axes = plt.subplots(n_dopings, n_temperatures, figsize=figsize)

        for d, t in np.ndindex(self.fermi_levels.shape):
            if n_dopings == 1 and n_temperatures == 1:
                ax = axes
            elif n_dopings == 1:
                ax = axes[t]
            elif n_temperatures == 1:
                ax = axes[d]
            else:
                ax = axes[d, t]

            if d == n_dopings - 1 and t == n_temperatures - 1:
                show_legend = True
            else:
                show_legend = False

            title = "n = {:.2g} cm$^{{-3}}$\t T = {}".format(
                self.doping[d] * (1 / bohr_to_cm) ** 3, self.temperatures[t]
            )

            _plot_rates_to_axis(
                ax,
                energies,
                rates[:, d, t],
                self.scattering_labels,
                self.fd_cutoffs,
                plot_total_rate=plot_total_rate,
                plot_fd_tols=plot_fd_tols,
                ymin=ymin,
                ymax=ymax,
                show_legend=show_legend,
                legend_kwargs=_legend_kwargs,
                normalize_energy=norm_e,
                title=title,
            )

        plt.subplots_adjust(wspace=0.3)

        return plt


def _plot_rates_to_axis(
    ax,
    energies,
    rates,
    labels,
    fd_cutoffs,
    plot_total_rate: bool = False,
    plot_fd_tols: bool = True,
    show_legend: bool = True,
    legend_kwargs=None,
    ymin=None,
    ymax=None,
    colors=None,
    normalize_energy=0,
    title=None,
):
    if colors is None:
        color_cycler = cycler(color=_seaborn_colors)
        ax.set_prop_cycle(color_cycler)

    if legend_kwargs is None:
        legend_kwargs = {}

    labels = deepcopy(labels)
    rates = deepcopy(rates)
    if plot_total_rate:
        # add total rates column
        rates = np.concatenate((rates, rates.sum(axis=0)[None, ...]))
        labels += ["total"]

    min_fd = fd_cutoffs[0]
    max_fd = fd_cutoffs[1]
    min_e = min_fd - (max_fd - min_fd) * 0.1
    max_e = max_fd + (max_fd - min_fd) * 0.1

    energies = energies.ravel()
    energy_mask = (energies > min_e) & (energies < max_e)
    energies = energies[energy_mask]

    plt_rates = {}
    for label, rate in zip(labels, rates):
        rate = rate.ravel()
        plt_rates[label] = rate[energy_mask]

    rates_in_cutoffs = np.array(list(plt_rates.values()))
    ymin, ymax = _get_rate_ylims(rates_in_cutoffs, ymin=ymin, ymax=ymax)

    # convert energies to eV and normalise
    norm_energies = energies / units.eV - normalize_energy
    norm_min_fd = min_fd / units.eV - normalize_energy
    norm_max_fd = max_fd / units.eV - normalize_energy
    norm_min_e = min_e / units.eV - normalize_energy
    norm_max_e = max_e / units.eV - normalize_energy

    for label, rate in plt_rates.items():
        ax.scatter(norm_energies, rate, label=label)

    if plot_fd_tols:
        ax.plot((norm_min_fd, norm_min_fd), (ymin, ymax), c="gray", ls="--", alpha=0.5)
        ax.plot(
            (norm_max_fd, norm_max_fd),
            (ymin, ymax),
            c="gray",
            ls="--",
            alpha=0.5,
            label="FD cutoffs",
        )

    ax.semilogy()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim((norm_min_e, norm_max_e))
    ax.set(xlabel="energy (eV)", ylabel="scattering rate (s$^-1$)", title=title)

    if show_legend:
        ax.legend(**legend_kwargs)


def _get_rate_ylims(
    rates, ymin: Optional[float] = None, ymax: Optional[float] = None, pad: float = 0.1
):
    rates = rates[np.isfinite(rates)]

    if not ymin:
        min_log = np.log10(np.min(rates))
    else:
        min_log = np.log10(ymin)

    if not ymax:
        max_log = np.log10(np.max(rates))
    else:
        max_log = np.log10(ymax)

    diff_log = max_log - min_log

    if not ymin:
        log_tmp = min_log - diff_log * pad
        ymin = 10 ** log_tmp

    if not ymax:
        log_tmp = max_log + diff_log * pad
        ymax = 10 ** log_tmp

    return ymin, ymax


def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma ** 2)
