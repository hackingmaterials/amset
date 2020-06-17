from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sumo.plotting import styled_plot

from amset.plot.base import BaseAmsetPlotter, amset_base_style

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

    @styled_plot(amset_base_style)
    def get_plot(
        self,
        plot_fd_tols: bool = False,
        plot_total_rate: bool = False,
        ymin: float = None,
        ymax: float = None,
        xmin: float = None,
        xmax: float = None,
        normalize_energy: bool = True,
        separate_rates: bool = True,
        doping_idx=0,
        temperature_idx=0,
        legend=True,
        style=None,
        no_base_style=False,
        fonts=None,
    ):
        if normalize_energy and self.is_metal:
            norm_e = self.fermi_levels[0][0]
        elif normalize_energy:
            cb_idx = {s: v + 1 for s, v in self.vb_idx.items()}
            norm_e = np.max([self.energies[s][: cb_idx[s]] for s in self.spins])
        else:
            norm_e = 0

        if doping_idx is None and len(self.doping) == 1:
            doping_idx = 0

        if temperature_idx is None and len(self.temperatures) == 1:
            temperature_idx = 0

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

        base_size = 3.2

        def get_size(nplots):
            width = base_size * nplots
            width += (
                base_size * 0.25 * (nplots - 1)
            )  # account for spacing between plots
            return width

        if ny == 1:
            figsize = (get_size(nx), base_size)
            fig, axes = plt.subplots(1, nx, figsize=figsize)
        else:
            figsize = (get_size(nx), get_size(ny))
            fig, axes = plt.subplots(ny, nx, figsize=figsize)

        for y_idx, p_x_idx in enumerate(p_idx):
            for x_idx, (n, t) in enumerate(p_x_idx):
                if nx == 1 and ny == 1:
                    ax = axes
                elif ny == 1:
                    ax = axes[x_idx]
                else:
                    ax = axes[y_idx, x_idx]

                if x_idx == nx - 1 and y_idx == ny - 1 and legend:
                    show_legend = True
                else:
                    show_legend = False

                str_doping = _latex_float(self.doping[n])

                title = r"n = ${}$ cm$^{{-3}}$    " "T = {} K".format(
                    str_doping, self.temperatures[t]
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
                    xmin=xmin,
                    xmax=xmax,
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
        xmin=None,
        xmax=None,
        normalize_energy=0,
        plot_lifetimes=False,
        scaling_factor=1,
    ):
        rates = self.plot_rates[:, doping_idx, temperature_idx]
        if separate_rates:
            labels = self.scattering_labels.tolist()
        else:
            rates = np.sum(rates, axis=0)[None, ...]
            labels = ["overall"]

        if legend_kwargs is None:
            legend_kwargs = {}

        labels = deepcopy(labels)
        if plot_total_rate:
            # add total rates column
            rates = np.concatenate((rates, np.sum(rates, axis=0)[None, ...]))
            labels += ["total"]

        min_fd = self.fd_cutoffs[0]
        max_fd = self.fd_cutoffs[1]
        if not xmin:
            min_e = min_fd
        else:
            min_e = xmin

        if not xmax:
            max_e = max_fd
        else:
            max_e = xmax

        energies = self.plot_energies.ravel()
        energy_mask = (energies > min_e) & (energies < max_e)
        energies = energies[energy_mask]

        plt_rates = {}
        for label, rate in zip(labels, rates):
            rate = rate.ravel()
            if plot_lifetimes:
                rate = 1 / rate
            rate *= scaling_factor
            plt_rates[label] = rate[energy_mask]

        rates_in_cutoffs = np.array(list(plt_rates.values()))

        ymin, ymax = _get_rate_ylims(rates_in_cutoffs, ymin=ymin, ymax=ymax)

        # convert energies to eV and normalise
        norm_energies = energies - normalize_energy
        norm_min_fd = min_fd - normalize_energy
        norm_max_fd = max_fd - normalize_energy
        norm_min_e = min_e - normalize_energy
        norm_max_e = max_e - normalize_energy

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
        scaling_string = "{:g} ".format(scaling_factor) if scaling_factor != 1 else ""
        if plot_lifetimes:
            ylabel = "Scattering lifetime ({}s)".format(scaling_string)
        else:
            ylabel = "Scattering rate ({}s$^{{-1}}$)".format(scaling_string)

        ax.set(xlabel="Energy (eV)", ylabel=ylabel)

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


def _latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} x 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
