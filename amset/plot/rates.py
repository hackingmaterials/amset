from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from BoltzTraP2 import units
from matplotlib import cycler

from amset.constants import bohr_to_cm
from amset.plot.base import BaseAmsetPlotter, seaborn_colors

_legend_kwargs = {"loc": "upper left", "bbox_to_anchor": (1, 1), "frameon": False}


class AmsetRatesPlotter(BaseAmsetPlotter):
    def get_plot(
        self,
        plot_fd_tols: bool = True,
        plot_total_rate: bool = False,
        ymin: float = None,
        ymax: float = None,
        normalize_energy: bool = True,
        separate_rates: bool = True,
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

            if separate_rates:
                plot_rates = rates[:, d, t]
                labels = self.scattering_labels

            else:
                plot_rates = np.sum(rates[:, d, t], axis=0)[None, ...]
                labels = ["overall"]

            _plot_rates_to_axis(
                ax,
                energies,
                plot_rates,
                labels,
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
        color_cycler = cycler(color=seaborn_colors)
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
