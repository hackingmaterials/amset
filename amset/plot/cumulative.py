import numpy as np
from sumo.plotting import pretty_plot

from amset.constants import bohr_to_m, bohr_to_nm, boltzmann_au, s_to_au
from amset.electronic_structure.fd import dfdde
from amset.plot import BaseMeshPlotter, amset_base_style, styled_plot

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


_y_labels = {"conductivity": r"Cumulative conductivity (\%)"}

_x_labels = {
    "mean free path": "Mean free path (nm)",
    "group velocity": "Group velocity (m/s)",
    "scattering rate": r"Scattering rate (s$^{-1}$)",
}

_conversions = {
    "mean free path": bohr_to_nm,
    "group velocity": bohr_to_m,
    "scattering rate": 1,
}


class CumulativePlotter(BaseMeshPlotter):
    @styled_plot(amset_base_style)
    def get_plot(
        self,
        n_idx,
        t_idx,
        x_property="mean free path",
        y_property="conductivity",
        height=6,
        width=6,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        logx=False,
        plt=None,
        style=None,
        no_base_style=False,
        fonts=None,
    ):
        x_values, y_values = self.get_plot_data(n_idx, t_idx, x_property, y_property)

        plt = pretty_plot(width=width, height=height, plt=plt)
        ax = plt.gca()
        ax.plot(x_values, y_values)

        xlabel = xlabel if xlabel else _x_labels[x_property.lower()]
        ylabel = ylabel if ylabel else _y_labels[y_property.lower()]
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)

        if logx:
            ax.semilogx()

    def get_plot_data(
        self, n_idx, t_idx, x_property="mean free path", y_property="conductivity"
    ):
        x_values = self._get_x_values(n_idx, t_idx, x_property)
        y_values = self._get_y_values(n_idx, t_idx, y_property)
        x_values, y_values = _get_cummulative_sum(x_values, y_values)

        x_values *= _conversions[x_property.lower()]
        y_values *= 100  # to percent

        return x_values, y_values

    def _get_y_values(self, n_idx, t_idx, y_property):
        if y_property.lower() == "conductivity":
            y_values = self._get_conductivity(n_idx, t_idx)
        else:
            raise ValueError(f"unknown y_property: {y_property}")

        return np.ravel(y_values)

    def _get_x_values(self, n_idx, t_idx, x_property):
        if x_property.lower() == "mean free path":
            x_values = self._get_mean_free_path(n_idx, t_idx)
        elif x_property.lower() == "group velocity":
            x_values = self._get_group_velocity()
        elif x_property.lower() == "scattering rate":
            x_values = self._get_scattering_rates(n_idx, t_idx)
        else:
            raise ValueError(f"unknown x_property: {x_property}")

        return np.ravel(x_values)

    def _get_conductivity(self, n_idx, t_idx):
        velocities = self._get_group_velocity()
        lifetimes = 1 / self._get_scattering_rates(n_idx, t_idx)

        energies = self._get_energies()
        _, weights = np.unique(self.ir_to_full_kpoint_mapping, return_counts=True)

        ef = self.fermi_levels[n_idx, t_idx]
        temp = self.temperatures[t_idx]

        dfde = -dfdde(energies, ef, temp * boltzmann_au)
        nkpoints = len(self.kpoints)

        integrand = velocities**2 * lifetimes * dfde * weights[None, :] / nkpoints
        conductivity = np.sum(integrand)

        return integrand / conductivity

    def _get_mean_free_path(self, n_idx, t_idx):
        # mean free path in bohr
        group_velocity = self._get_group_velocity()
        scattering_rate = self._get_scattering_rates(n_idx, t_idx)
        return group_velocity / scattering_rate

    def _get_group_velocity(self):
        # mean free path in bohr / s
        velocities = {s: np.linalg.norm(v, axis=2) for s, v in self.velocities.items()}
        all_velocities = np.concatenate([v for v in velocities.values()])
        all_velocities[all_velocities < 0.005] = 0.005  # handle very small velocities
        return all_velocities * s_to_au

    def _get_scattering_rates(self, n_idx, t_idx):
        rates = self.scattering_rates.values()
        return np.concatenate([np.sum(r[:, n_idx, t_idx], axis=0) for r in rates])

    def _get_energies(self):
        return np.concatenate([e for e in self.energies.values()])


def _get_cummulative_sum(x_values, y_values):
    sort_idx = np.argsort(x_values)
    y_values = y_values[sort_idx]
    x_values = x_values[sort_idx]
    cumsum = np.cumsum(y_values)
    return x_values, cumsum
