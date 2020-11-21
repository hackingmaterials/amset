import colorsys
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from amset.constants import boltzmann_ev
from amset.electronic_structure.fd import dfdde
from amset.plot import get_figsize, styled_plot
from amset.plot.base import BaseMeshPlotter, amset_base_style, base_total_color
from amset.plot.transport import fancy_format_doping, fancy_format_temp, get_lim

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


_legend_kwargs = {"loc": "upper left", "bbox_to_anchor": (1, 1), "frameon": False}
_fmt_data = {
    "rate": {
        "label": "Scattering rate",
        "label_symbol": r"$\tau^{-1}$",
        "unit": "s$^{-1}$",
    },
    "lifetime": {
        "label": "Scattering lifetime",
        "label_symbol": r"$\tau$",
        "unit": "s",
    },
    "v2tau": {"label": r"$v^2\tau$", "label_symbol": r"$v^2\tau$", "unit": r"cm$^2$/s"},
    "v2taudfde": {
        "label": r"$v^2\tau \left[ - \frac{\mathrm{d}f}{\mathrm{d}\varepsilon} \right ]$",
        "label_symbol": r"$v^2\tau \left[ - \frac{\mathrm{d}f}{\mathrm{d}\varepsilon} \right ]$",
        "unit": r"cm$^2$/s",
    },
    "energy": {"label": "Energy (eV)", "label_symbol": r"$\varepsilon$ (eV)"},
}


class RatesPlotter(BaseMeshPlotter):
    def __init__(self, *args, pad=0.1, use_symbol=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_energies = np.vstack([self.energies[s] for s in self.spins])
        self.plot_velocities = np.vstack([self.velocities[s] for s in self.spins])
        self.plot_norm_velocities = np.linalg.norm(self.plot_velocities, axis=-1)
        self.plot_rates = np.concatenate(
            [self.scattering_rates[s] for s in self.spins], axis=3
        )
        self.label_key = "label_symbol" if use_symbol else "label"
        self.pad = pad

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
        total_color=None,
        show_dfde=False,
        plot_type="rate",
        title=True,
        axes=None,
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

        if axes is None:
            _, axes = plt.subplots(ny, nx, figsize=get_figsize(ny, nx))

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

                doping_fmt = fancy_format_doping(self.doping[n])
                temp_fmt = fancy_format_temp(self.temperatures[t])
                title_str = f"{doping_fmt}    {temp_fmt}"

                self.plot_rates_to_axis(
                    ax,
                    n,
                    t,
                    plot_type=plot_type,
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
                    total_color=total_color,
                    show_dfde=show_dfde,
                )
                if title:
                    ax.set(title=title_str)

        return plt

    def plot_rates_to_axis(
        self,
        ax,
        doping_idx,
        temperature_idx,
        plot_type="rate",
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
        scaling_factor=1,
        total_color=None,
        show_dfde=False,
    ):
        rates = self.plot_rates[:, doping_idx, temperature_idx]
        if separate_rates:
            sort_idx = np.argsort(self.scattering_labels)
            labels = self.scattering_labels[sort_idx].tolist()
            rates = rates[sort_idx]
        else:
            rates = np.sum(rates, axis=0)[None, ...]
            labels = ["rates"]

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
            min_e = xmin + normalize_energy

        if not xmax:
            max_e = max_fd
        else:
            max_e = xmax + normalize_energy

        energies = self.plot_energies.ravel()
        energy_mask = (energies > min_e) & (energies < max_e)
        energies = energies[energy_mask]

        dfde_occ = -dfdde(
            energies,
            self.fermi_levels[doping_idx, temperature_idx],
            boltzmann_ev * self.temperatures[temperature_idx],
        )

        plt_rates = {}
        for label, rate in zip(labels, rates):
            rate = rate.ravel()[energy_mask]
            if plot_type == "lifetime":
                rate = 1 / rate
            elif plot_type == "v2tau":
                rate = self.plot_norm_velocities.ravel()[energy_mask] ** 2 / rate
            elif plot_type == "v2taudfde":
                rate = (
                    self.plot_norm_velocities.ravel()[energy_mask] ** 2
                    * dfde_occ
                    / rate
                )
            elif plot_type == "rate":
                pass
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

            rate *= scaling_factor
            plt_rates[label] = rate

        rates_in_cutoffs = np.array(list(plt_rates.values()))
        ylim = get_lim(
            rates_in_cutoffs[rates_in_cutoffs > 0], ymin, ymax, True, self.pad
        )

        # convert energies to eV and normalise
        norm_energies = energies - normalize_energy
        norm_min_fd = min_fd - normalize_energy
        norm_max_fd = max_fd - normalize_energy
        xlim = (min_e - normalize_energy, max_e - normalize_energy)

        if total_color is None:
            total_color = base_total_color

        colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (label, rate) in enumerate(plt_rates.items()):
            if label == "total":
                c = total_color
            else:
                c = colors[i]

            legend_c = c
            if show_dfde:
                c = _get_lightened_colors(c, dfde_occ / np.max(dfde_occ))
                # sort on lightness to put darker points on top of lighter ones
                sort_idx = np.argsort(dfde_occ)
                norm_energies_sort = norm_energies[sort_idx]
                rate = rate[sort_idx]
                c = c[sort_idx]
            else:
                norm_energies_sort = norm_energies

            ax.scatter(norm_energies_sort, rate, c=c, rasterized=True)
            # hack to plot get the correct label if using dfde occupation
            ax.scatter(-10000000, 0, c=legend_c, label=label)

        if plot_fd_tols:
            ax.plot((norm_min_fd, norm_min_fd), ylim, c="gray", ls="--", alpha=0.5)
            ax.plot(
                (norm_max_fd, norm_max_fd),
                ylim,
                c="gray",
                ls="--",
                alpha=0.5,
                label="FD cutoffs",
            )

        scaling_string = "{:g} ".format(scaling_factor) if scaling_factor != 1 else ""
        data_label = _fmt_data[plot_type][self.label_key]
        data_unit = _fmt_data[plot_type]["unit"]
        ylabel = f"{data_label} ({scaling_string}{data_unit})"
        xlabel = _fmt_data["energy"][self.label_key]

        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim, xlim=xlim)
        ax.semilogy()

        if show_legend:
            ax.legend(**legend_kwargs)


def _get_lightened_colors(color, occupation, min_saturation=0, min_lightness=0.92):
    color_rgb = np.array(to_rgb(color))
    h, ll, s = colorsys.rgb_to_hls(*color_rgb)

    lighten_amount = 1 - occupation
    s = s - (s - min_saturation) * lighten_amount
    ll = ll - (ll - min_lightness) * lighten_amount

    light_colors_rgb = []
    for s_new, l_new in zip(s, ll):
        light_colors_rgb.append(colorsys.hls_to_rgb(h, l_new, s_new))
    return np.array(light_colors_rgb)
