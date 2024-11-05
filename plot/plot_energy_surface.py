from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
from matplotlib import cm


def plot_1d_thermo(distance: List, temps: List, energies, filename: str = 'G_surface.png'):
    """
    Plot the 1D PES at different temperatures
    Args:
        distance:
        temps:
        energies:
        filename:

    Returns: None

    """
    point_matrix, temp_matrix = np.meshgrid(distance, temps)

    # finding the highest free energy path
    ts_distance = []
    for fe in energies:
        ts_distance.append(distance[list(fe).index(np.max(fe))])

    _ = plt.figure(figsize=(5, 4))
    _ = plt.contour(point_matrix, temp_matrix, energies, levels=len(temps) + 1,
                    colors='black', linestyles='dashed', linewidths=0.5)
    CS = plt.contourf(point_matrix, temp_matrix, energies, levels=len(temps), cmap="RdBu_r")
    plt.scatter(ts_distance, temps, marker="o", s=30, alpha=1, zorder=10, facecolors="white",
                edgecolors="black")
    plt.plot(ts_distance, temps, ls="--", alpha=0.8, c='black')
    plt.colorbar(CS, label='G / $(kcal \cdot mol^{-1})$')
    plt.ylabel('Temperature / $K$')
    plt.xlabel('Distance / $\AA$')

    plt.tight_layout()
    plt.savefig(filename, dpi=500) if filename is not None else plt.show()

    plt.close('all')

    return None


class Smooth:
    """
    Smooth plotting space
    """
    def __init__(self, interp_factor: int = 4):
        self.interp_factor = interp_factor

    def smooth1d(self, x: List[float], values):
        new_arr = np.linspace(np.min(x), np.max(x), num=self.interp_factor * len(x))
        interp_func = interp1d(x, values)
        return new_arr, interp_func(new_arr)

    def smooth2d(self, x: List[float], y: List[float], values, kx: int = 3, ky: int = 3):
        new_x = np.linspace(np.min(x), np.max(x), num=self.interp_factor * len(x))
        new_y = np.linspace(np.min(y), np.max(y), num=self.interp_factor * len(y))
        interp_func = self._spline_2d(x, y, values, kx=kx, ky=ky)
        return new_x, new_y, interp_func(new_x, new_y)

    def interp_2d_point(self,
                        x: List[float],
                        y: List[float],
                        values,
                        point: Tuple[float, float],
                        kx: int = 3,
                        ky: int = 3):
        interp_func = self._spline_2d(x, y, values, kx=kx, ky=ky)
        return point[0], point[1], interp_func(*point)

    def interp_2d_func(self,
                       x: List[float],
                       y: List[float],
                       values,
                       kx: int = 3,
                       ky: int = 3):
        interp_func = self._spline_2d(x, y, values, kx=kx, ky=ky)
        return interp_func

    @staticmethod
    def _spline_2d(r_x, r_y, value, kx: int = 3, ky: int = 3):
        """
        Spline the surface using Scipy. As of scipy v1.7.1 RectBivariateSpline
        can only accept monotonically increasing arrays. This function thus
        reverses arrays and the energies when appropriate, so the spline can
        be fit.

        -----------------------------------------------------------------------
        Returns:
            (scipy.interpolate.RectBivariateSpline): Spline
        """

        if r_x[0] < r_x[-1] and r_y[0] < r_y[-1]:
            # Both x and y are strictly increasing functions
            return RectBivariateSpline(r_x, r_y, value, kx=kx, ky=ky)

        if r_x[0] > r_x[-1] and r_y[0] < r_y[-1]:
            # Swap x order to get strictly increasing in both dims
            return RectBivariateSpline(r_x[::-1], r_y, value[::-1, :], kx=kx, ky=ky)

        if r_x[0] < r_x[-1] and r_y[0] > r_y[-1]:
            # or with y
            return RectBivariateSpline(r_x, r_y[::-1], value[:, ::-1], kx=kx, ky=ky)

        # Reverse both the x and y arrays
        return RectBivariateSpline(
            r_x[::-1], r_y[::-1], value[::-1, ::-1], kx=kx, ky=ky
        )


def plot_2d(r_x,
            r_y,
            energies,
            energy_units_name: str = 'kcal mol-1',
            interp_factor: int = 0,
            ts_coor: Tuple[float, float] = None,
            filename: str = "energy_surface.png",
            calc_relative_value: bool = True) -> None:
    """
    Plot the PES in two dimensions

    -----------------------------------------------------------------------
    Arguments:
        r_x:
        r_y:
        energies:
        interp_factor:
        energy_units_name:
        ts_coor: coordinate of transition state
        filename:
        calc_relative_value: calculate relative value
    """

    if interp_factor > 0:
        smooth = Smooth(interp_factor=interp_factor)
        r_x, r_y, energies = smooth.smooth2d(r_x, r_y, energies)
    if ts_coor is not None:
        smooth = Smooth()
        ts_x, ts_y, ts_energy = smooth.interp_2d_point(r_x, r_y, energies, point=ts_coor)
    else:
        ts_x, ts_y, ts_energy = None, None, None

    # Set up the figure and axes to plot the 3D and projected surfaces on
    _ = plt.figure(figsize=(10, 6))
    ax0 = plt.subplot(1, 2, 1, projection=Axes3D.name)
    ax1 = plt.subplot(1, 2, 2)

    # Convert the energies in the 2D array from the base Hartree units
    if calc_relative_value:
        energies = energies - np.min(energies)

    ax0.plot_surface(
        *np.meshgrid(r_x, r_y), energies.T, cmap=plt.get_cmap("RdBu_r")
    )
    if ts_coor is not None:
        ax0.scatter(ts_x, ts_y, ts_energy, marker='*', color="black")
    ax0.set_xlabel("$r_1$ / Å")
    ax0.set_ylabel("$r_2$ / Å")
    ax0.set_zlabel(f"$E$ / {energy_units_name}")

    im = ax1.imshow(
        energies.T,
        aspect=(abs(max(r_x)-min(r_x)) / abs(max(r_y)-min(r_y))),
        extent=(r_x[0], r_x[-1], r_y[0], r_y[-1]),
        origin="lower",
        cmap=plt.get_cmap("RdBu_r"),
    )
    if ts_coor is not None:
        ax1.scatter(ts_x, ts_y, marker='*', color="black")

    contour = ax1.contour(
        *np.meshgrid(r_x, r_y),
        energies.T,
        levels=8,
        origin="lower",
        colors="k",
        linewidths=1,
        alpha=0.5,
    )

    plt.clabel(contour, inline=1, fontsize=10, colors="k")

    cbar = plt.colorbar(im, fraction=0.0458, pad=0.04)
    cbar.set_label(f"$E$ / {energy_units_name}")
    ax1.set_xlabel("$r_1$ / Å")
    ax1.set_ylabel("$r_2$ / Å")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(filename, dpi=500) if filename is not None else plt.show()
    plt.close()

    return None


def get_xy_value(x_value, y_value, surface, contour_height: List[float], interp_factor: int = 4):
    """
    Get x or y value of
    Args:
        x_value:
        y_value:
        surface:
        contour_height:
        interp_factor:

    Returns:

    """
    assert np.max(surface) > np.max(contour_height) and np.min(surface) < np.min(contour_height), "Maximum(minim) " \
                                                        "value of contour_height are higher(lower) than surface value"
    if interp_factor > 0:
        x_len, y_len = len(x_value), len(y_value)
        kx = x_len - 1 if x_len <= 3 else 3
        ky = y_len - 1 if y_len <= 3 else 3
        smooth = Smooth(interp_factor=interp_factor)
        x_value, y_value, surface = smooth.smooth2d(x_value, y_value, surface, kx=kx, ky=ky)

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    c = ax.contour(x_value, y_value, surface.T, contour_height)
    lines = c.allsegs
    plt.close(fig)

    return {h: line for h, line in zip(contour_height, lines)}

