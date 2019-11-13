from copy import deepcopy
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from BoltzTraP2 import units
from matplotlib import cycler

from amset.data import AmsetData

_seaborn_colors = [
    (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
    (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
    (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
    (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    (0.8, 0.7254901960784313, 0.4549019607843137),
    (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
]
_legend_kwargs = {"loc": "upper left", "bbox_to_anchor": (1, 1), "frameon": False}


class AmsetPlotter(object):

    def __init__(self, amset_data: AmsetData):
        # TODO: Check we have all the data we need
        self._amset_data = amset_data

    def plot_rates(
        self,
        plot_fd_tols: bool = True,
        plot_total_rate: bool = False,
        ymin: float = None,
        ymax: float = None,
        normalize_energy: bool = True,
    ):
        if normalize_energy and self._amset_data.is_metal:
            norm_e = self._amset_data.intrinsic_fermi_level
        elif normalize_energy:
            vb_idx = self._amset_data.vb_idx
            spins = self._amset_data.spins
            norm_e = np.max(
                [self._amset_data.energies[s][: vb_idx[s] + 1] for s in spins]
            )
        else:
            norm_e = 0

        norm_e /= units.eV

        energies = np.vstack(list(self._amset_data.energies.values()))
        rates = np.concatenate(list(self._amset_data.scattering_rates.values()), axis=3)

        n_dopings = len(self._amset_data.doping)
        n_temperatures = len(self._amset_data.temperatures)
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

        for d, t in np.ndindex(self._amset_data.fermi_levels.shape):
            if len(self._amset_data.doping) == 1:
                ax = axes[t]
            elif len(self._amset_data.temperatures) == 1:
                ax = axes[d]
            else:
                ax = axes[d, t]

            if d == n_dopings - 1 and t == n_temperatures - 1:
                show_legend = True
            else:
                show_legend = False

            title = "n = {:.2g} cm$^{{-3}}$\t T = {}".format(
                self._amset_data.doping[d],
                self._amset_data.temperatures[t]
            )

            _plot_rates_to_axis(
                ax,
                energies,
                rates[:, d, t],
                self._amset_data.scattering_labels,
                self._amset_data.fd_cutoffs,
                plot_total_rate=plot_total_rate,
                plot_fd_tols=plot_fd_tols,
                ymin=ymin,
                ymax=ymax,
                show_legend=show_legend,
                legend_kwargs=_legend_kwargs,
                normalize_energy=norm_e,
                title=title
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
