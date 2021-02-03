import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from glob import glob


def scalar_prepare(df):
    # df.Strength = (df.Strength - df.Strength.min()) / (df.Strength.max() - df.Strength.min())
    Z = df.pivot_table(index='Longitude', columns='Latitude', values='Strength').T.values
    X_unique = np.sort(df.Longitude.unique())  # probably need to flip these?
    Y_unique = np.sort(df.Latitude.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    return X, Y, Z


def vector_prepare(df):
    X = df.Longitude.to_numpy()
    Y = df.Latitude.to_numpy()

    # df.Strength = df.Strength / df.Strength.max()
    # vector calculation
    df.Azimuth = df.Azimuth.apply(np.deg2rad)
    U = (df.Strength * df.Azimuth.apply(np.sin)).to_numpy()
    V = (df.Strength * df.Azimuth.apply(np.cos)).to_numpy()

    # normalise
    # U = (U - U.min()) / (U.range())
    return X, Y, U, V


def main():
    files = glob('data/xyz/*.xyz')
    # stations = pd.read_csv('data/stations.csv', index_col=0)

    # pi anisotropy
    ans_2v = pd.read_csv(files[2], header=None, delim_whitespace=True,
                         names=['Latitude', 'Longitude', 'Azimuth', 'Strength'])
    # isotropic speed
    iso_v = pd.read_csv(files[0], header=None, delim_whitespace=True, names=['Latitude', 'Longitude', 'Strength'])

    iso_v = scalar_prepare(iso_v)
    ans_2v = vector_prepare(ans_2v)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    plt.title('Anisotropy of the Arabian Plate')
    # boundingbox = (49, 60, 21.5, 28.5)  # (x0, x1, y0, y1)
    boundingbox = [15, 75, 2, 55]  # (x0, x1, y0, y1)

    ax.set_extent(boundingbox)

    filled_c = ax.contourf(*iso_v, 200, transform=proj, cmap='seismic_r')
    # line_c = ax.contour(*iso_v, levels=4, colors=['black'],transform=ccrs.Miller())
    # ax.scatter(iso_v.Latitude.to_numpy(), iso_v.Longitude.to_numpy(), c=iso_v.Strength.to_numpy())
    aniso = ax.quiver(*ans_2v, transform=proj, pivot='mid', headlength=0, headaxislength=0, units='inches', scale=2.54, scale_units='inches')
    ax.quiverkey(aniso, X=0.3, Y=0.9, U=1, label='2% peak-to-peak anisotropy')
    # sta = ax.scatter(stations.Longitude, stations.Latitude)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.stock_img()
    # ax.set_global()

    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    fig.colorbar(filled_c, orientation='vertical', cax=cax)

    plt.show()


if __name__ == '__main__':
    main()
