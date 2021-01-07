from pathlib import Path

import numpy as np

from amset.plot.base import PlotData, write_plot_data


def test_plot_data(clean_dir):
    # test initialization
    plot_data = PlotData(
        x_property="doping",
        y_property="cond",
        x_label="Carrier concentration [cm$^{3}$]",
        y_label="Conductivity [S/m]",
        labels=["T = 100 K", "T = 200 K"],
        x=np.array([1e16, 1e17, 1e18]),
        y=np.array([[1, 2, 3], [4, 5, 6]]),
    )

    # test comment
    assert "x-axis: Carrier concentration" in plot_data.comment
    assert "y-axis: Conductivity" in plot_data.comment
    assert "T=100K" in plot_data.comment

    # test data
    assert plot_data.data.shape == (3, 3)
    assert plot_data.data[0, 1] == 1e17
    assert plot_data.data[2, 1] == 5

    # test writing to file
    filename = Path("test_data.dat")
    plot_data.to_file(filename)
    assert filename.exists()
    assert "x-axis: Carrier concentration" in filename.read_text()


def test_write_plot_data(clean_dir):
    # test initialization
    plot_data = [
        PlotData(
            x_property="doping",
            y_property="cond",
            x_label="Carrier concentration [cm$^{3}$]",
            y_label="Conductivity [S/m]",
            labels=["T = 100 K", "T = 200 K"],
            x=np.array([1e16, 1e17, 1e18]),
            y=np.array([[1, 2, 3], [4, 5, 99]]),
        ),
        PlotData(
            x_property="doping",
            y_property="seebeck",
            x_label="Carrier concentration [cm$^{3}$]",
            y_label="Seebeck coefficient [S/m]",
            labels=["T = 100 K", "T = 200 K"],
            x=np.array([1e16, 1e17, 1e18]),
            y=np.array([[10, 11, 12], [15, 16, 101]]),
        ),
    ]

    files = [
        {
            "filename": "cond.dat",
            "should_contain": ["y-axis: Conductivity", " 9.90000000e+01"],
        },
        {
            "filename": "seebeck.dat",
            "should_contain": ["y-axis: Seebeck coefficient", " 1.01000000e+02"],
        },
    ]

    # basic write
    write_plot_data(plot_data)
    for file_data in files:
        filename = Path(file_data["filename"])
        assert filename.exists(), f"{filename} does not exist"

        contents = filename.read_text()
        for expected in file_data["should_contain"]:
            assert expected in contents, f"{filename} does not contain {expected}"

    # write with prefix
    prefix = "test_"
    write_plot_data(plot_data, prefix=prefix)
    for file_data in files:
        filename = Path(prefix + file_data["filename"])
        assert filename.exists(), f"{filename} does not exist"

        contents = filename.read_text()
        for expected in file_data["should_contain"]:
            assert expected in contents, f"{filename} does not contain {expected}"
