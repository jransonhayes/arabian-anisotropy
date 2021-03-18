"""
for plotting loads of datasets in once go, using my file structure.
"""
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from glob import glob
from plot import Settings, scalar_prepare, vector_prepare
import os


def plot(*args):
    period = args[0]
    mode = args[1]
    ax = args[2]
    settings = Settings(mode, period)

    if mode == 'final':
        path = os.path.expanduser(f'~/Desktop/final/{period}/')
    elif mode == 'synth-iso':
        path = os.path.expanduser(f'~/Desktop/synthresults/iso/{period}/')
    elif mode == 'synth-aniso':
        path = os.path.expanduser(f'~/Desktop/synthresults/aniso/{period}/')
    else:
        path = ''

    def plot_iso(path):
        path = glob(path + '*_v_*.xyz')[0]
        iso_v = pd.read_csv(path, header=None, delim_whitespace=True, names=['Latitude', 'Longitude', 'Strength']).sample(10)
        iso_v = scalar_prepare(iso_v)
        filled_c = ax.contourf(*iso_v, 200, transform=ccrs.PlateCarree(), cmap='seismic_r')
        return filled_c

    def plot_aniso(path):
        path = glob(path + '*_an2v_*.xyz')[0]

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
                                 central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2))
    ax.set_title(settings.title)
    # boundingbox = (49, 60, 21.5, 28.5)  # (x0, x1, y0, y1)

    ax.set_extent(boundingbox, crs=ccrs.PlateCarree())

    if settings.draw_iso:
        filled_c = plot_iso(path)

        cax = fig.add_axes([ax.get_position().x1 + 0.03, ax.get_position().y0, 0.01, 1.1*ax.get_position().height])
        fig.colorbar(filled_c, orientation='vertical', cax=cax)

    if settings.draw_aniso:
        aniso, quivkey = plot_aniso(path)

    # line_c = ax.contour(*iso_v, levels=4, colors=['black'],transform=ccrs.Miller())
    # ax.scatter(iso_v.Latitude.to_numpy(), iso_v.Longitude.to_numpy(), c=iso_v.Strength.to_numpy())

    # sta = ax.scatter(stations.Longitude, stations.Latitude)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.stock_img()
    # ax.set_global()

    # plt.show()


if __name__ == '__main__':
    boundingbox = [30, 65, 8, 50]  # (x0, x1, y0, y1)
    proj = ccrs.LambertConformal(central_longitude=(boundingbox[0] + (boundingbox[1] - boundingbox[0]) / 2),
                                 central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, subplot_kw={'projection': proj},
                            figsize=(6.25, 3.25), squeeze=True)




    for period, ax in zip([10, 18], axs.ravel()):
        plot(period, 'final', ax)

    # fig.subplots_adjust(wspace=1)
    fig.tight_layout()

    plt.show()
