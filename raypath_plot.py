"""
for plotting raypath coverage plots. Draws lines for each station pair. Can be run for all periods or a single period
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from formatter import read_raw_file
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot(period, ax):
    df = read_raw_file(period)

    # boundingbox = [25, 71, 8, 50]  # (x0, x1, y0, y1)
    # proj = ccrs.LambertConformal(central_longitude=(boundingbox[0] + (boundingbox[1] - boundingbox[0]) / 2),
    #                              central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2),
    #                              standard_parallels=(15, 40))

    ax.set_extent(boundingbox, crs=ccrs.PlateCarree())

    # ax.scatter(df.lon_sta1, df.lat_sta1, transform=ccrs.PlateCarree())
    # ax.scatter(df.lon_sta2, df.lat_sta2, transform=ccrs.PlateCarree())

    for row in df.itertuples():
        xs = [row[4], row[2]]  # lon2, lon1
        ys = [row[3], row[1]]  # lat2, lat1

        ax.plot(xs, ys, c=m.to_rgba(row[5]), alpha=0.01, transform=ccrs.PlateCarree())

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title(str(period) + 's', fontsize=30)

    # plt.savefig(f'raypath_coverage/{period}.png')
    return ax


def main():
    proj = ccrs.LambertConformal(central_longitude=(boundingbox[0] + (boundingbox[1] - boundingbox[0]) / 2),
                                 central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2),
                                 standard_parallels=(15, 40))

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, subplot_kw={'projection': proj},
                            figsize=(35, 26), squeeze=False)

    deraveled = axs.ravel()
    fig.delaxes(deraveled[8])
    fig.delaxes(deraveled[6])

    pairings = {i[1]: i[0] for i in enumerate([10, 18, 25, 36, 46, 55, 70])}
    pairings[70] = 7
    for period in [10, 18, 25, 36, 46, 55, 70]:
        ax = deraveled[pairings[period]]
        plot(period, ax)

    fig.tight_layout()
    fig.suptitle('Raypath Coverage', fontsize=50)
    fig.colorbar(m)
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == '__main__':
    boundingbox = [25, 71, 8, 50]
    norm = Normalize(vmin=1.9, vmax=4.4)
    cmap = cm.plasma
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    main()
