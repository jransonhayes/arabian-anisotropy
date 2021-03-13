import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from glob import glob


class Settings:
    def __init__(self, mode, period=None, extra=None):
        """
        :param mode: ('synth iso', 'synth-aniso', 'final')
        """
        if mode == 'synth-iso':
            self.skip = 2
            self.scale = 7
            self.draw_iso = True
            self.draw_aniso = True
            self.title = f'Synthetic Isotropic Velocity Test'

            if period:
                self.title += f': {period}s'

        elif mode == 'synth-aniso':
            self.skip = 2
            self.scale = 7  # 7
            self.draw_iso = True
            self.draw_aniso = True
            self.title = 'Synthetic Anisotropy Test'

            if period:
                self.title += f": {period}s"

        elif mode == 'final':
            self.skip = 2
            self.scale = 2.54
            self.draw_iso = True
            self.draw_aniso = True
            self.title = f'Period: {period}s'

            if extra:
                self.title += ", " + extra
        else:
            raise ValueError


def scalar_prepare(df):
    # df.Strength = (df.Strength - df.Strength.min()) / (df.Strength.max() - df.Strength.min())
    Z = df.pivot_table(index='Longitude', columns='Latitude', values='Strength').T.values
    X_unique = np.sort(df.Longitude.unique())  # probably need to flip these?
    Y_unique = np.sort(df.Latitude.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    return X, Y, Z


def vector_prepare(df):
    df = df[df.Strength > 0.1]  # remove 0 vectors
    X = df.Longitude.to_numpy()
    Y = df.Latitude.to_numpy()

    # df.Strength = df.Strength / df.Strength.max()
    # vector calculation
    df.Azimuth = df.Azimuth.apply(np.deg2rad)
    U = (df.Strength * df.Azimuth.apply(np.sin)).to_numpy()
    V = (df.Strength * df.Azimuth.apply(np.cos)).to_numpy()

    # normalise
    # U = (U - U.min()) / (U.range())
    return np.array([X, Y, U, V])


def main():
    settings = Settings('synth-aniso')

    def plot_iso():
        path = glob('data/xyz/*_v_*.xyz')[0]
        iso_v = pd.read_csv(path, header=None, delim_whitespace=True, names=['Latitude', 'Longitude', 'Strength'])
        iso_v = scalar_prepare(iso_v)
        filled_c = ax.contourf(*iso_v, 200, transform=ccrs.PlateCarree(), cmap='seismic_r')
        return filled_c

    def plot_aniso():
        path = glob('data/xyz/*_an2v_*.xyz')[0]

        ans_2v = pd.read_csv(path, header=None, delim_whitespace=True,
                             names=['Latitude', 'Longitude', 'Azimuth', 'Strength'])
        ans_2v = vector_prepare(ans_2v)

        if not ans_2v.size:
            return None, None

        skip = slice(None, None, settings.skip)
        ans_2v = ans_2v[:, skip]  # reduce vector density

        aniso = ax.quiver(*ans_2v, transform=ccrs.PlateCarree(), pivot='mid', headlength=0, headaxislength=0,
                          units='inches',
                          scale=settings.scale,  # 2.54 for 1 cm : 100 %
                          scale_units='inches',
                          width=0.03)
        quivkey = ax.quiverkey(aniso, X=0.5, Y=-0.1, U=1, label='2% peak-to-peak anisotropy')
        return aniso, quivkey

    # stations = pd.read_csv('data/stations.csv', index_col=0)

    # pi anisotropy

    # isotropic speed

    boundingbox = [30, 65, 8, 50]  # (x0, x1, y0, y1)
    proj = ccrs.LambertConformal(central_longitude=(boundingbox[0] + (boundingbox[1] - boundingbox[0]) / 2),
                                 central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2),
                                 standard_parallels=(15, 40))
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    plt.title(settings.title)
    # boundingbox = (49, 60, 21.5, 28.5)  # (x0, x1, y0, y1)

    ax.set_extent(boundingbox, crs=ccrs.PlateCarree())

    if settings.draw_iso:
        filled_c = plot_iso()

        # filled_c.set_clim(2.05, 3.85) # use constant colorbar range

        cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(filled_c, orientation='vertical', cax=cax)

    if settings.draw_aniso:
        aniso, quivkey = plot_aniso()

    # line_c = ax.contour(*iso_v, levels=4, colors=['black'],transform=ccrs.Miller())
    # ax.scatter(iso_v.Latitude.to_numpy(), iso_v.Longitude.to_numpy(), c=iso_v.Strength.to_numpy())

    # sta = ax.scatter(stations.Longitude, stations.Latitude)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.stock_img()
    # ax.set_global()

    plt.show()


if __name__ == '__main__':
    main()
