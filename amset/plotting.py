import os

import numpy as np

from amset.utils.general import AmsetError


def get_amset_plots(amset, k_plots=None, e_plots=None, mobility=True,
                    mode='offline', concentrations='all', temperatures='all',
                    carrier_types=None, direction=None, fontsize=30,
                    ticksize=25, path=None, dir_name="plots", margins=100,
                    fontfamily="serif", include_avg=False, **kwargs):
    """
    Plots the given k_plots and E_plots properties.

    Args:
        amset (Amset): Amset class that has been run.
        k_plots ([str]): the names of the quantities to be plotted against
            norm(k) options: 'energy', 'df0dk', 'velocity', or 'all' (not in a
            list) to plot everything
        e_plots ([str]) the names of the quantities to be plotted against
            E options: 'frequency', 'relaxation time', '_all_elastic', 'df0dk',
            'velocity', 'ACD', 'IMP', 'PIE', 'g', 'g_POP', 'g_th', 'S_i', 'S_o',
            or 'all' (not in a list) to plot everything
        mobility (bool): if True, create a mobility against temperature plot
        mode (str): plotly mode defaulting to 'offline'; see figrecipe
            documentation for more information. For example if "static"
            (saving the figures directly w/o interactive display), the
            plotly credentials are required.
        concentrations ([float]): carrier concentrations, or the string 'all' to
            plot the results of calculations done with all input concentrations
        temperatures ([int]): temperatures to be included in the plots
        carrier_types (list of strings): select carrier types to plot data for -
            ['n'], ['p'], or ['n', 'p']
        direction (list of strings): options to include in list are 'x', 'y',
            'z', 'avg'; determines which components of vector quantities are
            plotted
        fontsize (int): size of title and axis label text
        ticksize (int): size of axis tick label text
        path (string): location to save plots
        dir_name (str): the name of the folder where plot files are saved
        margins (int): figrecipes plotly margins
        fontfamily (string): plotly font
        include_avg (bool): whether to include the "average" mobility in
            the mobility plot. False is recommended as mixing "overall" and
            "average" mobility may be confusing.
        **kwargs: other keyword arguments of matminer.figrecipes.plot.PlotlyFig
            for example, for setting plotly credential when mode=="static"
            note that if mode is "static" Plotly username and api_key need
            to be set
    """
    k_plots = k_plots or []
    e_plots = e_plots or []
    carrier_types = carrier_types or ['n', 'p']
    direction = direction or ['avg']
    path = os.path.join(path or amset.calc_dir, dir_name)

    if not os.path.exists(path):
        os.makedirs(path)

    supported_k_plots = ['energy', 'df0dk', 'velocity'] + amset.elastic_scats
    supported_e_plots = ['frequency', 'relaxation time', 'df0dk',
                         'velocity'] + amset.elastic_scats

    if "POP" in amset.inelastic_scats:
        supported_e_plots += ['g', 'g_POP', 'g_th', 'S_i', 'S_o']
        supported_k_plots += ['g', 'g_POP', 'g_th', 'S_i', 'S_o']
    if k_plots == 'all':
        k_plots = supported_k_plots
    if e_plots == 'all':
        e_plots = supported_e_plots
    if concentrations == 'all':
        concentrations = amset.dopings
    if temperatures == 'all':
        temperatures = amset.temperatures

    # make copies of mutable arguments
    concentrations = [int(c) for c in concentrations]
    carrier_types = list(carrier_types)
    direction = list(direction)

    mu_list = ["overall"] + amset.elastic_scats + amset.inelastic_scats
    if include_avg:
        mu_list.insert(1, "average")
    # separate temperature dependent and independent properties
    all_temp_independent_k_props = ['energy', 'velocity']
    all_temp_independent_e_props = ['frequency', 'velocity']

    temp_independent_k_props = []
    temp_independent_e_props = []

    temp_dependent_k_props = []
    for prop in k_plots:
        if prop not in supported_k_plots:
            raise AmsetError(amset.logger,
                             'No support for {} vs. k plot!'.format(prop))
        if prop in all_temp_independent_k_props:
            temp_independent_k_props.append(prop)
        else:
            temp_dependent_k_props.append(prop)

    temp_dependent_e_props = []
    for prop in e_plots:
        if prop not in supported_e_plots:
            raise AmsetError(amset.logger,
                             'No support for {} vs. E plot!'.format(prop))
        if prop in all_temp_independent_e_props:
            temp_independent_e_props.append(prop)
        else:
            temp_dependent_e_props.append(prop)

    vec = {'energy': False,
           'velocity': True,
           'frequency': False}

    for tp in carrier_types:
        x_data = {'k': amset.kgrid0[tp]["norm(k)"][0],
                  'E': [E - amset.cbm_vbm[tp]["energy"]
                        for E in amset.egrid0[tp]["energy"]]}
        x_axis_label = {'k': 'norm(k)', 'E': 'energy (eV)'}

        # plots of scalar properties first
        for x_value, y_values in [('k', temp_independent_k_props),
                                  ('E', temp_independent_e_props)]:
            y_data_temp_independent = {
                'k': {'energy': amset.kgrid0[tp]['energy'][0],
                      'velocity': amset.kgrid0[tp]["norm(v)"][0]},
                'E': {'frequency': amset.Efrequency0[tp]}
            }

            for y_value in y_values:
                if not vec[y_value]:
                    title = None
                    if y_value == 'frequency':
                        title = 'Energy Histogram for {}'.format(
                            amset.tp_title[tp])
                    create_plots(
                        x_axis_label[x_value], y_value, tp, tp, fontsize,
                        ticksize, path, margins, fontfamily,
                        plot_data=[(x_data[x_value],
                                    y_data_temp_independent[x_value][y_value])],
                        mode=mode, x_label_short=x_value, title=title, **kwargs)

        for direc in direction:
            y_data_temp_independent = {
                'k': {'energy': amset.kgrid0[tp]['energy'][0],
                      'velocity': amset.kgrid0[tp]["norm(v)"][0]},
                'E': {'frequency': amset.Efrequency0[tp],
                      'velocity': [amset.get_scalar_output(p, direc)
                                   for p in amset.egrid0[tp]['velocity']]}}

            tp_dir = tp + '_' + direc

            # temperature independent k and E plots: energy(k), velocity(k),
            # histogram(E), velocity(E)
            for x_value, y_values in [('k', temp_independent_k_props),
                                      ('E', temp_independent_e_props)]:
                for y_value in y_values:
                    if vec[y_value]:
                        create_plots(
                            x_axis_label[x_value], y_value, tp, tp_dir,
                            fontsize, ticksize, path, margins, fontfamily,
                            plot_data=[(
                                x_data[x_value],
                                y_data_temp_independent[x_value][y_value])],
                            mode=mode, x_label_short=x_value, **kwargs)

            # want variable of the form:
            # y_data_temp_dependent[k or E][prop][temp]
            # (the following lines reorganize)
            try:
                y_data_temp_dependent = {
                    'k': {kp: {} for kp in temp_dependent_k_props},
                    'E': {ep: {} for ep in temp_dependent_e_props}}

                for c in concentrations:
                    for T in temperatures:
                        for kprop in temp_dependent_k_props:
                            y_data_temp_dependent['k'][kprop][(c, T)] = [
                                amset.get_scalar_output(p, direc)
                                for p in amset.kgrid0[tp][kprop][c][T][0]]

                        for eprop in temp_dependent_e_props:
                            y_data_temp_dependent['E'][eprop][(c, T)] = [
                                amset.get_scalar_output(p, direc)
                                for p in amset.egrid0[tp][eprop][c][T]]

            except KeyError:  # for when from_file is called
                y_data_temp_dependent = {
                    'k': {kp: {} for kp in temp_dependent_k_props},
                    'E': {ep: {} for ep in temp_dependent_e_props}}

                for c in concentrations:
                    for T in temperatures:
                        for kprop in temp_dependent_k_props:
                            y_data_temp_dependent['k'][kprop][(c, T)] = [
                                amset.get_scalar_output(p, direc) for p in
                                amset.kgrid0[tp][kprop][str(c)][str(int(T))][0]]

                        for eprop in temp_dependent_e_props:
                            y_data_temp_dependent['E'][eprop][(c, T)] = [
                                amset.get_scalar_output(p, direc) for p in
                                amset.egrid0[tp][eprop][str(c)][str(int(T))]]

            # temperature dependent k and E plots
            for x_value, y_values in [('k', temp_dependent_k_props),
                                      ('E', temp_dependent_e_props)]:
                for y_value in y_values:
                    plot_data = []
                    names = []
                    for c in concentrations:
                        for T in temperatures:
                            plot_data.append((
                                x_data[x_value],
                                y_data_temp_dependent[x_value][y_value][(
                                    c, T)]))
                            names.append('c={0:.1e}, T={1} K'.format(c, T))

                    create_plots(x_axis_label[x_value], y_value, tp, tp_dir,
                                 fontsize, ticksize, path, margins, fontfamily,
                                 plot_data=plot_data, mode=mode,
                                 x_label_short=x_value, names=names, **kwargs)

            # mobility plots as a function of temperature (the only plot that
            # does not have k or E on the x axis)
            if mobility:
                for c in concentrations:
                    plot_data = []
                    names = []
                    mo_mags = []
                    for mo in mu_list:
                        try:
                            mo_values = [amset.mobility[tp][mo][c][T]
                                         for T in amset.temperatures]
                        except KeyError:  # for when from_file is called
                            mo_values = [amset.mobility[tp][mo][str(int(c))][
                                             str(int(T))]
                                         for T in amset.temperatures]
                        plot_data.append((
                            amset.temperatures,
                            [amset.get_scalar_output(mo_value, direc)
                             for mo_value in mo_values]))
                        names.append(mo)
                        mo_mags.extend([np.log10(1+abs(np.mean(m)))
                                        for m in mo_values])

                    scale = None
                    if np.max(mo_mags) - np.min(mo_mags) > 1.5:
                        scale = 'log'
                    create_plots(
                        "Temperature (K)", "Mobility (cm2/V.s)", tp, tp_dir,
                        fontsize-5, ticksize-5, path, margins, fontfamily,
                        plot_data=plot_data, mode=mode, names=names,
                        xy_modes='lines+markers', y_label_short="mobility",
                        y_axis_type=scale,
                        title="{0}-type mobility at c={1:.2e}".format(tp, c),
                        **kwargs)


def create_plots(x_title, y_title, tp,
                 file_suffix, fontsize, ticksize, path, margins, fontfamily,
                 plot_data, mode='offline', names=None, labels=None,
                 x_label_short='', y_label_short=None, xy_modes='markers',
                 y_axis_type='linear', title=None, empty_markers=True,
                 **kwargs):
    """
    A wrapper function with args mostly consistent with
    matminer.figrecipes.plot.PlotlyFig

    Args:
        x_title (str): label of the x-axis
        y_title (str): label of the y-axis
        tp (str): "n" or "p"
        file_suffix (str): small suffix for filename (NOT a file format)
        fontsize (int):
        ticksize (int):
        path (str): root folder where the plot will be saved.
        margins (float or [float]): figrecipe PlotlyFig margins
        fontfamily (str):
        plot_data ([(x_data, y_data) tuples]): the actual data to be plotted
        mode (str): plot mode. "offline" and "static" recommended. "static"
            would automatically set the file format to .png
        names ([str]): names of the traces
        labels ([str]): the labels of the scatter points
        x_label_short (str): used for distinguishing filenames
        y_label_short (str):  used for distinguishing filenames
        xy_modes (str): mode of the xy scatter plots: "markers", "lines+markers"
        y_axis_type (str): e.g. "log" for logscale
        title (str): the title of the plot appearing at the top
        empty_markers (bool): whether the markers are empty (filled if False)
        **kwargs: other keyword arguments of matminer.figrecipes.plot.PlotlyFig
                for example, for setting plotly credential when mode=="static"

    Returns (None): to return the dict

    """
    from matminer.figrecipes.plot import PlotlyFig
    plot_data = list(plot_data)
    marker_symbols = range(44)
    if empty_markers:
        marker_symbols = [i+100 for i in marker_symbols]
    tp_title = {"n": "conduction band(s)", "p": "valence band(s)"}
    if title is None:
        title = '{} for {}'.format(y_title, tp_title[tp])
    if y_label_short is None:
        y_label_short = y_title
    if not x_label_short:
        filename = os.path.join(path, "{}_{}".format(
            y_label_short, file_suffix))
    else:
        filename = os.path.join(path, "{}_{}_{}".format(
            y_label_short, x_label_short, file_suffix))
    if mode == "static":
        if not filename.endswith(".png"):
            filename += ".png"
    pf = PlotlyFig(x_title=x_title, y_title=y_title, y_scale=y_axis_type,
                   title=title, fontsize=fontsize,
                   mode=mode, filename=filename, ticksize=ticksize,
                   margins=margins, fontfamily=fontfamily, **kwargs)
    pf.xy(plot_data, names=names, labels=labels, modes=xy_modes,
          marker_scale=1.1, markers=[{'symbol': marker_symbols[i],
                                      'line': {'width': 2, 'color': 'black'}}
                                     for i, _ in enumerate(plot_data)])
