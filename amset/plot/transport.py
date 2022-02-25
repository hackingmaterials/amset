import warnings

import matplotlib.pyplot
import numpy as np
from matplotlib import cm

from amset.plot import (
    BaseTransportPlotter,
    amset_base_style,
    get_figsize,
    pretty_subplot,
    styled_plot,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

from amset.plot.base import PlotData

property_data = {
    "conductivity": {
        "label": "Conductivity (S/m)",
        "label_symbol": r"$\sigma$ (S/m)",
        "log": True,
    },
    "seebeck": {
        "label": r"Seebeck coefficient ($\mu$V/K)",
        "label_symbol": r"$S$ ($\mu$V/K)",
        "log": False,
    },
    "thermal conductivity": {
        "label": "Thermal conductivity (W/mK)",
        "label_symbol": r"$\kappa_e$ (W/mK)",
        "log": True,
    },
    "mobility": {
        "label": "Mobility (cm$^2$/Vs)",
        "label_symbol": r"$\mu$ (cm$^2$/Vs)",
        "log": False,
    },
    "power factor": {
        "label": r"Power factor (mW$\,$m$^{-1}$$\,$K$^{-2}$)",
        "label_symbol": r"PF (mW$\,$m$^{-1}$$\,$K$^{-2}$)",
        "log": False,
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
dir_mapping = {0: "xx", 1: "yy", 2: "zz"}
ls_mapping = {0: "-", 1: "--", 2: ":"}


class TransportPlotter(BaseTransportPlotter):
    def __init__(self, data, use_symbol=False, pad=0.05, average=True):
        super().__init__(data)
        self.label_key = "label_symbol" if use_symbol else "label"
        self.pad = pad
        self.average = average

    @styled_plot(amset_base_style)
    def get_plot(
        self,
        properties=("conductivity", "seebeck", "thermal conductivity"),
        x_property=None,
        doping_idx=None,
        temperature_idx=None,
        grid=None,
        height=None,
        width=None,
        doping_type=None,
        conductivity_label=None,
        conductivity_min=None,
        conductivity_max=None,
        log_conductivity=None,
        seebeck_label=None,
        seebeck_min=None,
        seebeck_max=None,
        log_seebeck=None,
        mobility_label=None,
        mobility_min=None,
        mobility_max=None,
        log_mobility=None,
        thermal_conductivity_label=None,
        thermal_conductivity_min=None,
        thermal_conductivity_max=None,
        log_thermal_conductivity=None,
        power_factor_label=None,
        power_factor_min=None,
        power_factor_max=None,
        log_power_factor=None,
        xlabel=None,
        xmin=None,
        xmax=None,
        logx=None,
        legend=True,
        axes=None,
        return_plot_data=False,
        plt=None,
        style=None,
        no_base_style=False,
        fonts=None,
    ):

        if axes is None:
            if grid is None or len(grid) == 0:
                grid = (1, len(properties))

            if np.product(grid) < len(properties):
                raise ValueError("Grid is not large enough to plot all properties")

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

            if np.product(grid) < len(properties):
                raise ValueError("Not enough axes to plot all properties")

        if doping_type is not None and doping_idx is not None:
            warnings.warn(
                "Both doping type and doping indexes have been set. This can cause "
                "unexpected behaviour."
            )

        prop_dict = {
            "conductivity": {
                "ylabel": conductivity_label,
                "ymin": conductivity_min,
                "ymax": conductivity_max,
                "logy": log_conductivity,
            },
            "mobility": {
                "ylabel": mobility_label,
                "ymin": mobility_min,
                "ymax": mobility_max,
                "logy": log_mobility,
            },
            "seebeck": {
                "ylabel": seebeck_label,
                "ymin": seebeck_min,
                "ymax": seebeck_max,
                "logy": log_seebeck,
            },
            "thermal conductivity": {
                "ylabel": thermal_conductivity_label,
                "ymin": thermal_conductivity_min,
                "ymax": thermal_conductivity_max,
                "logy": log_thermal_conductivity,
            },
            "power factor": {
                "ylabel": power_factor_label,
                "ymin": power_factor_min,
                "ymax": power_factor_max,
                "logy": log_power_factor,
            },
        }

        if np.product(grid) > len(properties):
            n_missing = np.product(grid) - len(properties)
            properties = list(properties) + [None] * n_missing

        plot_data = []
        for (x, y), prop in zip(np.ndindex(grid), properties):
            if grid[0] == 1 and grid[1] == 1:
                ax = axes
            elif grid[0] == 1:
                ax = axes[y]
            elif grid[1] == 1:
                ax = axes[x]
            else:
                ax = axes[x, y]

            if prop is None:
                ax.axis("off")

            else:
                prop_data = self.plot_property(
                    ax,
                    prop,
                    x_property=x_property,
                    temperature_idx=temperature_idx,
                    doping_idx=doping_idx,
                    doping_type=doping_type,
                    xmin=xmin,
                    xmax=xmax,
                    xlabel=xlabel,
                    logx=logx,
                    **prop_dict[prop],
                )
                plot_data.append(prop_data)

        if legend:
            if grid[0] == 1 and grid[1] == 1:
                ax = axes
            elif grid[0] == 1:
                ax = axes[-1]
            elif grid[1] == 1:
                ax = axes[0]
            else:
                ax = axes[0, -1]
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1)

        if return_plot_data:
            return matplotlib.pyplot, plot_data

        return matplotlib.pyplot

    def plot_property(
        self,
        ax,
        prop,
        x_property=None,
        temperature_idx=None,
        doping_idx=None,
        doping_type=None,
        ylabel=None,
        ymin=None,
        ymax=None,
        logy=None,
        xlabel=None,
        xmin=None,
        xmax=None,
        logx=None,
    ):
        if temperature_idx is None:
            temperature_idx = np.arange(self.temperatures.shape[0])
        elif isinstance(temperature_idx, int):
            temperature_idx = [temperature_idx]
        temperatures = self.temperatures[temperature_idx]

        if doping_idx is None:
            doping_idx = np.arange(self.doping.shape[0])
        elif isinstance(doping_idx, int):
            doping_idx = [doping_idx]

        if doping_type == "n":
            doping_idx = [i for i in doping_idx if self.doping[i] <= 0]
        elif doping_type == "p":
            doping_idx = [i for i in doping_idx if self.doping[i] >= 0]

        doping = self.doping[doping_idx]

        data = self._get_data(prop)

        if x_property is None and len(temperatures) == 1:
            x_property = "doping"
        elif x_property is None:
            x_property = "temperature"

        y = data[doping_idx][:, temperature_idx]
        if self.average:
            y = y.reshape((len(doping), len(temperatures)))  # make sure y is still 2D
        else:
            y = y.reshape((len(doping), len(temperatures), 3))
        if x_property == "doping":
            labels = [fancy_format_temp(t) for t in temperatures]
            x = doping
            y = np.swapaxes(y, 0, 1)
        elif x_property == "temperature":
            labels = [fancy_format_doping(d) for d in doping]
            x = temperatures
            y = y
        else:
            raise ValueError(f"Unknown x_property: {x_property}")

        if np.any((x > 0) & (x < 0)):
            raise ValueError(
                "Cannot plot n- and p- type properties on the same figure."
                "Use doping-idx to select doping indices"
            )
        x = np.abs(x)

        cmap = cm.get_cmap("viridis_r")
        colors = cmap(np.linspace(0.03, 1 - 0.03, len(y)))

        line_data = []
        for yi, label, color in zip(y, labels, colors):
            if self.average:
                lines = ax.plot(x, yi, label=label, c=color)
                line_data.extend(lines)
            else:
                for i in range(3):
                    lines = ax.plot(
                        x,
                        yi[:, i],
                        label=f"{dir_mapping[i]} {label}",
                        c=color,
                        ls=ls_mapping[i],
                    )
                    line_data.extend(lines)

        ylabel = ylabel if ylabel else property_data[prop][self.label_key]
        xlabel = xlabel if xlabel else property_data[x_property][self.label_key]
        logy = logy if logy is not None else property_data[prop]["log"]
        logx = logx if logx is not None else property_data[x_property]["log"]
        ylim = get_lim(y, ymin, ymax, logy, self.pad)
        xlim = get_lim(x, xmin, xmax, logx, self.pad)

        if logy:
            ax.semilogy()
        if logx:
            ax.semilogx()
        ax.set(ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim)

        return PlotData(
            x_property=x_property,
            y_property=prop,
            x_label=xlabel,
            y_label=ylabel,
            labels=[line.get_label() for line in line_data],
            x=x,
            y=np.array([line.get_ydata() for line in line_data]),
        )

    def _get_data(self, prop):
        if prop == "conductivity":
            data = self.conductivity
        elif prop == "seebeck":
            data = self.seebeck
        elif prop == "thermal conductivity":
            data = self.electronic_thermal_conductivity
        elif prop == "mobility":
            data = self.mobility
        elif prop == "power factor":
            data = self.seebeck**2 * self.conductivity * 1e-9  # convert to mW/(m K^2)
        else:
            raise ValueError(f"Unrecognised property: {prop}")

        if self.average:
            data = np.average(np.linalg.eigvalsh(data), axis=-1)
        else:
            data = np.diagonal(data, axis1=-2, axis2=-1)
        return data


def get_lim(data, vmin, vmax, logv, pad):
    data = data[np.isfinite(data)]
    data_min = np.log10(data.min()) if logv else data.min()
    data_max = np.log10(data.max()) if logv else data.max()
    data_pad = pad * (data_max - data_min)

    if vmin is None:
        vmin = data_min - data_pad
        if logv:
            vmin = 10**vmin

    if vmax is None:
        vmax = data_max + data_pad
        if logv:
            vmax = 10**vmax

    return vmin, vmax


def carrier_type(doping):
    if doping < 0:
        return "n"
    else:
        return "p"


def fancy_format_doping(doping):
    if carrier_type(doping) == "n":
        doping = abs(doping)
        prefix = "$n_e$"
    else:
        prefix = "$n_h$"
    return f"{prefix} = {format_doping(doping)}"


def fancy_format_temp(temperature):
    return f"T = {format_temp(temperature)}"


def format_temp(temperature):
    if temperature % 10 == 0:
        return f"{int(temperature)} K"
    else:
        return f"{temperature:.1f} K"


def format_doping(doping):
    log_doping = np.log10(doping)
    if log_doping % 1 == 0:
        return f"10$^{{{int(log_doping)}}}$ cm$^{{-3}}$"
    else:
        log_part = int(np.floor(log_doping))
        front_part = 10 ** (log_doping - log_part)
        return rf"{front_part:.2f}$\times$10$^{{{log_part}}}$ cm$^{{-3}}$"
