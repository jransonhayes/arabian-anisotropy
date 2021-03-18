"""
colours map by azimuth of anisotropy fast direction.
"""
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
from scipy import interpolate

def plot(period, **kwargs):
    # initial setup
    interp = kwargs.get('interp')
    path = os.path.expanduser(f'~/Desktop/final/{period}/')


    boundingbox = [30, 65, 8, 50]  # (x0, x1, y0, y1)
    proj = ccrs.LambertConformal(central_longitude=(boundingbox[0] + (boundingbox[1] - boundingbox[0]) / 2),
                                 central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2),
                                 standard_parallels=(15, 40))
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    path = glob(path + '*_an2v_*.xyz')[0]
    ans_2v = pd.read_csv(path, header=None, delim_whitespace=True,
                         names=['Latitude', 'Longitude', 'Azimuth', 'Strength'])

    def reorient(x):
        # reorients a vector from 0-360 to -90 to 90 degrees
        val = x % 360
        if 90 < val < 270:
            return val - 180
        if val >= 270:
            return val - 360
        else:
            return val

    ans_2v = ans_2v[ans_2v.Strength > 0.2]  # remove 0 vectors
    # ans_2v.Azimuth = ans_2v.Azimuth.apply(reorient)  # collapse orientations

    # grid data in the right format for contourf
    Z = ans_2v.pivot_table(index='Longitude', columns='Latitude', values='Azimuth').T.values
    X_unique = np.sort(ans_2v.Longitude.unique())
    Y_unique = np.sort(ans_2v.Latitude.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)


    # Z = np.sin(np.deg2rad(Z))

    # interpolation
    if interp:
        X = np.linspace(boundingbox[0], boundingbox[1], 1000)
        Y = np.linspace(boundingbox[2], boundingbox[3], 1000)
        Z = interpolate.griddata((ans_2v[['Longitude', 'Latitude']].to_numpy()), ans_2v.Azimuth,
                                 (X[None, :], Y[:, None]), method='cubic')

    Z = np.vectorize(reorient)(Z)
    cont = ax.contourf(X, Y, Z, 7, transform=ccrs.PlateCarree(), cmap='twilight_shifted')

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    # fig.subplots_adjust(hspace=0, left=0.01, right=0, wspace=0)
    fig.colorbar(cont, orientation='horizontal', fraction=.03)
    # cont.set_clim(vmin=-90, vmax=90)
    ax.set_title(str(period) + 's')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot(55, interp=True)
