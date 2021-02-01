import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
import numpy as np
import geopandas as gpd
from glob import glob


def main():
    stations = pd.read_csv('data/stations.csv', index_col=0)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    plt.title('Station Coverage')

    # boundingbox = (49, 60, 21.5, 28.5)  # (x0, x1, y0, y1)
    boundingbox = [15, 75, 2, 55]  # (x0, x1, y0, y1)
    ax.set_extent(boundingbox)

    xy = np.vstack([stations.Longitude, stations.Latitude])
    z = gaussian_kde(xy)(xy)

    ax.coastlines()
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    sta = ax.scatter(stations.Longitude, stations.Latitude, c=z, cmap='plasma')

    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.stock_img()
    ax.gridlines(draw_labels=True)

    cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    fig.colorbar(sta, orientation='vertical', cax=cax)
    # ax.set_global()

    plt.show()


if __name__ == '__main__':
    main()
