import warnings

import matplotlib.pyplot
import numpy as np
from matplotlib import cm

from amset.plot import amset_base_style, get_figsize, pretty_subplot, styled_plot
from amset.plot.base import BaseMultiTransportPlotter
from amset.plot.transport import (
    fancy_format_doping,
    fancy_format_temp,
    get_lim,
    property_data,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


class ConvergencePlotter(BaseMultiTransportPlotter):
    def __init__(self, data, use_symbol=False, pad=0.05):
        super().__init__(data)
        self.label_key = "label_symbol" if use_symbol else "label"
        self.pad = pad

    @styled_plot(amset_base_style)
    def get_plot(
        self,
        doping_idx=None,
        temperature_idx=None,
        properties=("conductivity", "seebeck", "thermal conductivity"),
        labels=None,
        x_property=None,
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

        if temperature_idx is None and doping_idx is None:
            if len(self.doping) == 1 or x_property == "temperature":
                warnings.warn(
                    "Either one of temperature-idx or doping-idx should be "
                    "set. Using doping-idx of 0"
                )
                if doping_type:
                    if doping_type == "n":
                        doping_idx = [i for i, d in enumerate(self.doping) if d <= 0]
                    else:
                        doping_idx = [i for i, d in enumerate(self.doping) if d >= 0]
                    if len(doping_idx) == 0:
                        raise ValueError(
                            f"{doping_type}-type doping specified but no doping "
                            f"concentrations of this type are in the calculations"
                        )
                    else:
                        doping_idx = doping_idx[0]
                else:
                    doping_idx = [0]
            else:
                warnings.warn(
                    "Either one of temperature-idx or doping-idx should be "
                    "set. Using temperature-idx of 0"
                )
                temperature_idx = [0]

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

        title = None
        for (x, y), prop in zip(np.ndindex(grid), properties):
            if grid[0] == 1 and grid[1] == 1:
                ax = axes
            elif grid[0] == 1:
                ax = axes[y]
            elif grid[1] == 1:
                ax = axes[x]
            else:
                ax = axes[x, y]

            title = self.plot_property(
                ax,
                prop,
                x_property=x_property,
                doping_type=doping_type,
                temperature_idx=temperature_idx,
                doping_idx=doping_idx,
                xmin=xmin,
                xmax=xmax,
                xlabel=xlabel,
                logx=logx,
                labels=labels,
                **prop_dict[prop],
            )

        if legend:
            if grid[0] == 1 and grid[1] == 1:
                ax = axes
            elif grid[0] == 1:
                ax = axes[-1]
            elif grid[1] == 1:
                ax = axes[0]
            else:
                ax = axes[0, -1]
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, title=title)

        return matplotlib.pyplot

    def plot_property(
        self,
        ax,
        prop,
        x_property=None,
        doping_type=None,
        temperature_idx=None,
        doping_idx=None,
        ylabel=None,
        ymin=None,
        ymax=None,
        logy=None,
        xlabel=None,
        xmin=None,
        xmax=None,
        logx=None,
        labels=None,
    ):
        if temperature_idx is None and doping_idx is None:
            raise ValueError("Either one of temperature-idx or doping-idx must be set")

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

        if len(doping_idx) != 1 and len(temperature_idx) != 1:
            raise ValueError("A specific temperature or doping index must be chosen")

        data = self._get_data(prop)

        if x_property is None and len(temperatures) == 1:
            x_property = "doping"
        elif x_property is None:
            x_property = "temperature"

        y = np.squeeze(data[:, doping_idx][:, :, temperature_idx])

        if x_property == "doping":
            annotation = fancy_format_temp(temperatures[0])
            x = doping
        elif x_property == "temperature":
            annotation = fancy_format_doping(doping[0])
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

        if labels is None:
            labels = [str(i) for i in range(self.n)]
        elif len(labels) != self.n:
            raise ValueError("Number of labels does not match number of data")

        cmap = cm.get_cmap("viridis_r")
        colors = cmap(np.linspace(0.05, 1, len(y)))

        for yi, label, color in zip(y, labels, colors):
            ax.plot(x, yi, label=label, c=color)

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
        return annotation

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

        data = np.linalg.eigvalsh(data)
        data = np.average(data, axis=-1)
        return data
