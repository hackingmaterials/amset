from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

from amset.constants import bohr_to_cm
from amset.plot.base import BaseAmsetPlotter, seaborn_colors
from BoltzTraP2 import units

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

_legend_kwargs = {"loc": "upper left", "bbox_to_anchor": (1, 1), "frameon": False}


class RatesPlotter(BaseAmsetPlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_energies = np.vstack([self.energies[s] for s in self.spins])
        self.plot_rates = np.concatenate(
            [self.scattering_rates[s] for s in self.spins], axis=3
        )

    def get_plot(
        self,
        plot_fd_tols: bool = True,
        plot_total_rate: bool = False,
        ymin: float = None,
        ymax: float = None,
        normalize_energy: bool = True,
        separate_rates: bool = True,
        doping_idx=0,
        temperature_idx=0,
        legend=True,
    ):
        if normalize_energy and self.is_metal:
            norm_e = self.fermi_levels[0][0]
        elif normalize_energy:
            cb_idx = {s: v + 1 for s, v in self.vb_idx.items()}
            norm_e = np.max([self.energies[s][: cb_idx[s]] for s in self.spins])
        else:
            norm_e = 0

        if doping_idx is None and temperature_idx is None:
            ny = len(self.doping)
            nx = len(self.temperatures)
            p_idx = [[(x, y) for y in range(nx)] for x in range(ny)]
        elif doping_idx is None:
            nx = len(self.doping)
            ny = 1
            p_idx = [[(x, temperature_idx) for x in range(nx)]]
        elif temperature_idx is None:
            nx = len(self.temperatures)
            ny = 1
            p_idx = [[(doping_idx, x) for x in range(nx)]]
        else:
            nx = 1
            ny = 1
            p_idx = [[(doping_idx, temperature_idx)]]

        norm_e /= units.eV
        base_size = 4

        if ny == 1:
            figsize = (1.2 * base_size * nx, base_size)
            fig, axes = plt.subplots(1, nx, figsize=figsize)
        else:
            figsize = (1.2 * base_size * nx, base_size * ny)
            fig, axes = plt.subplots(ny, nx, figsize=figsize)

        for y_idx, p_x_idx in enumerate(p_idx):
            for x_idx, (n, t) in enumerate(p_x_idx):
                if nx == 1 and ny == 1:
                    ax = axes
                elif ny == 1:
                    ax = axes[x_idx]
                else:
                    ax = axes[x_idx, y_idx]

                if x_idx == nx - 1 and y_idx == ny - 1 and legend:
                    show_legend = True
                else:
                    show_legend = False

                title = "n = {:.2g} cm$^{{-3}}$\t T = {}".format(
                    self.doping[n] * (1 / bohr_to_cm) ** 3, self.temperatures[t]
                )

                self.plot_rates_to_axis(
                    ax,
                    n,
                    t,
                    separate_rates=separate_rates,
                    plot_total_rate=plot_total_rate,
                    plot_fd_tols=plot_fd_tols,
                    ymin=ymin,
                    ymax=ymax,
                    show_legend=show_legend,
                    legend_kwargs=_legend_kwargs,
                    normalize_energy=norm_e,
                )
                ax.set(title=title)

        return plt

    def plot_rates_to_axis(
        self,
        ax,
        doping_idx,
        temperature_idx,
        separate_rates=True,
        plot_total_rate: bool = False,
        plot_fd_tols: bool = True,
        show_legend: bool = True,
        legend_kwargs=None,
        ymin=None,
        ymax=None,
        colors=None,
        normalize_energy=0,
    ):
        rates = self.plot_rates[:, doping_idx, temperature_idx]
        if separate_rates:
            labels = self.scattering_labels
        else:
            rates = np.sum(rates, axis=0)[None, ...]
            labels = ["overall"]

        if colors is None:
            color_cycler = cycler(color=seaborn_colors)
            ax.set_prop_cycle(color_cycler)

        if legend_kwargs is None:
            legend_kwargs = {}

        labels = deepcopy(labels)
        if plot_total_rate:
            # add total rates column
            rates = np.concatenate((rates, rates.sum(axis=0)[None, ...]))
            labels += ["total"]

        min_fd = self.fd_cutoffs[0]
        max_fd = self.fd_cutoffs[1]
        min_e = min_fd - (max_fd - min_fd) * 0.1
        max_e = max_fd + (max_fd - min_fd) * 0.1

        energies = self.plot_energies.ravel()
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
            ax.plot(
                (norm_min_fd, norm_min_fd), (ymin, ymax), c="gray", ls="--", alpha=0.5
            )
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
        ax.set(xlabel="energy (eV)", ylabel="scattering rate (s$^-1$)")

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
