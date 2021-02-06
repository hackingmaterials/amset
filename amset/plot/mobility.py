import warnings

import matplotlib
import numpy as np

from amset.plot import get_figsize, pretty_subplot, styled_plot
from amset.plot.base import BaseTransportPlotter, amset_base_style, base_total_color
from amset.plot.transport import (
    dir_mapping,
    fancy_format_doping,
    fancy_format_temp,
    get_lim,
    ls_mapping,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


_property_data = {
    "mobility": {
        "label": "Mobility (cm$^2$/Vs)",
        "label_symbol": r"$\mu$ (cm$^2$/Vs)",
        "log": True,
    },
    "temperature": {
        "label": "Temperature (K)",
        "label_symbol": "$T$ (K)",
        "log": False,
    },
    "doping": {
        "label": "Carrier concentration (cm$^{-3}$)",
        "label_symbol": "$n$ (cm$^{-3}$)",
        "log": True,
    },
}


class MobilityPlotter(BaseTransportPlotter):
    def __init__(
        self, data, use_symbol=False, pad=0.05, average=True, separate_mobility=True
    ):
        super().__init__(data)
        self.label_key = "label_symbol" if use_symbol else "label"
        self.pad = pad
        self.average = average
        self.separate_mobility = separate_mobility

    @styled_plot(amset_base_style)
    def get_plot(
        self,
        x_property=None,
        doping_idx=None,
        temperature_idx=None,
        doping_type=None,
        grid=None,
        height=None,
        width=None,
        ylabel=None,
        ymin=None,
        ymax=None,
        logy=None,
        xlabel=None,
        xmin=None,
        xmax=None,
        logx=None,
        legend=True,
        title=True,
        total_color=None,
        axes=None,
        style=None,
        no_base_style=False,
        fonts=None,
    ):
        if doping_type is not None and doping_idx is not None:
            warnings.warn(
                "Both doping type and doping indexes have been set. This can cause "
                "unexpected behaviour."
            )

        if temperature_idx is None:
            temperature_idx = np.arange(self.temperatures.shape[0])
        elif isinstance(temperature_idx, int):
            temperature_idx = [temperature_idx]

        if doping_idx is None:
            doping_idx = np.arange(self.doping.shape[0])
        elif isinstance(doping_idx, int):
            doping_idx = [doping_idx]

        if doping_type == "n":
            doping_idx = [i for i in doping_idx if self.doping[i] <= 0]
        elif doping_type == "p":
            doping_idx = [i for i in doping_idx if self.doping[i] >= 0]

        if x_property is None and len(temperature_idx) == 1:
            x_property = "doping"
        elif x_property is None:
            x_property = "temperature"

        if x_property == "doping":
            if grid is None or len(grid) == 0:
                grid = (1, len(temperature_idx))
            primary_idxs = temperature_idx
            secondary_idxs = doping_idx
        elif x_property == "temperature":
            if grid is None or len(grid) == 0:
                grid = (1, len(doping_idx))
            primary_idxs = doping_idx
            secondary_idxs = temperature_idx
        else:
            raise ValueError(f"Unrecognised x_property: {x_property}")

        if axes is None:
            width, height = get_figsize(*grid, width=width, height=height)
            _, axes = pretty_subplot(*grid, width=width, height=height)
        else:
            if not isinstance(axes, (list, tuple, np.ndarray)):
                # single axis given
                grid = (1, 1)
            else:
                axes = np.array(axes)
                if grid is None:
                    grid = axes.shape
                    if len(grid) == 1:
                        grid = (1, grid[0])

            if len(primary_idxs) > np.product(grid):
                raise ValueError(
                    "Axes array not commensurate with doping_idx and temperature_idx"
                )

        for (x, y), primary_idx in zip(np.ndindex(grid), primary_idxs):
            if grid[0] == 1 and grid[1] == 1:
                ax = axes
            elif grid[0] == 1:
                ax = axes[y]
            elif grid[1] == 1:
                ax = axes[x]
            else:
                ax = axes[x, y]

            self.plot_mobility(
                ax,
                x_property,
                primary_idx,
                secondary_idxs,
                xlabel=xlabel,
                xmin=xmin,
                xmax=xmax,
                logx=logx,
                ylabel=ylabel,
                ymin=ymin,
                ymax=ymax,
                logy=logy,
                title=title,
                total_color=total_color,
            )
            ax.set()

        if legend and (self.separate_mobility or not self.average):
            if grid[0] == 1 and grid[1] == 1:
                ax = axes
            elif grid[0] == 1:
                ax = axes[-1]
            elif grid[1] == 1:
                ax = axes[0]
            else:
                ax = axes[0, -1]
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1)

        return matplotlib.pyplot

    def plot_mobility(
        self,
        ax,
        x_property,
        primary_idx,
        secondary_idxs,
        xlabel=None,
        xmin=None,
        xmax=None,
        logx=None,
        ylabel=None,
        ymin=None,
        ymax=None,
        logy=None,
        title=True,
        total_color=None,
    ):
        if x_property == "temperature":
            labels, y = self._get_data(primary_idx, secondary_idxs)
            x = self.temperatures[secondary_idxs]
            title_str = fancy_format_doping(self.doping[primary_idx])
        elif x_property == "doping":
            labels, y = self._get_data(secondary_idxs, primary_idx)
            x = self.doping[secondary_idxs]
            if np.any(x < 0) and np.any(x > 0):
                warnings.warn(
                    "You are plotting both n- and p-type carrier concentrations on the "
                    "same figure. Try using the --n-type and --p-type options instead."
                )
            x = np.abs(self.doping[secondary_idxs])
            title_str = fancy_format_temp(self.temperatures[primary_idx])
        else:
            raise ValueError(f"Unrecognised x_property: {x_property}")

        for i, (yi, label) in enumerate(zip(y, labels)):
            label = label.replace("overall", "total")
            if label == "total":
                color = base_total_color if total_color is None else total_color
            else:
                color = f"C{i}"

            if self.average:
                ax.plot(x, yi, label=label, c=color)
            else:
                for j in range(3):
                    ax.plot(
                        x,
                        yi[:, j],
                        label=f"{dir_mapping[j]} {label}",
                        c=color,
                        ls=ls_mapping[j],
                    )

        ylabel = ylabel if ylabel else _property_data["mobility"][self.label_key]
        xlabel = xlabel if xlabel else _property_data[x_property][self.label_key]
        logy = logy if logy is not None else _property_data["mobility"]["log"]
        logx = logx if logx is not None else _property_data[x_property]["log"]
        ylim = get_lim(y, ymin, ymax, logy, self.pad)
        xlim = get_lim(x, xmin, xmax, logx, self.pad)

        if logy:
            ax.semilogy()
        if logx:
            ax.semilogx()

        if not title:
            title_str = None

        ax.set(ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim, title=title_str)

    def _get_data(self, n_idx, t_idx):
        if self.separate_mobility:
            labels = sorted(set(self.mechanisms) - {"overall"}) + ["overall"]
            data = np.array([self.get_mobility(m)[n_idx, t_idx] for m in labels])
        else:
            labels = ["mobility"]
            data = self.mobility[n_idx, t_idx][None]  # add dummy first axis

        if self.average:
            data = np.average(np.linalg.eigvalsh(data), axis=-1)
        else:
            data = np.diagonal(data, axis1=-2, axis2=-1)
        return labels, data
