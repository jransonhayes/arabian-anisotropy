import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

period = 18

# path = os.path.expanduser(f'~/Desktop/final/{period}/')
path = os.path.expanduser(f'~/Desktop/Masters/synthetic results/{period}/')

path = glob(path + '*_an2v_*.xyz')[0]
ans_2v = pd.read_csv(path, header=None, delim_whitespace=True, names=['Latitude', 'Longitude', 'Azimuth', 'Strength'])


def vector_prepare(df):
    X = df.Longitude.to_numpy()
    Y = df.Latitude.to_numpy()

    # df.Strength = df.Strength / df.Strength.max()
    # vector calculation
    df.Azimuth = df.Azimuth.apply(np.deg2rad)
    U = (df.Strength * df.Azimuth.apply(np.sin)).to_numpy()
    V = (df.Strength * df.Azimuth.apply(np.cos)).to_numpy()
    C = df.Cluster
    # normalise
    # U = (U - U.min()) / (U.range())
    return np.array([X, Y, U, V, C])


def plot_quiver(df):
    ans_2v_gridded = vector_prepare(df)
    ans_2v_gridded = ans_2v_gridded[:, slice(None, None, 2)]
    fig = plt.figure(figsize=(12, 8))
    boundingbox = [30, 65, 8, 50]  # (x0, x1, y0, y1)

    proj = ccrs.LambertConformal(central_longitude=(boundingbox[0] + (boundingbox[1] - boundingbox[0]) / 2),
                                 central_latitude=(boundingbox[2] + (boundingbox[3] - boundingbox[2]) / 2),
                                 standard_parallels=(15, 40))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    aniso = ax.quiver(*ans_2v_gridded, transform=ccrs.PlateCarree(), pivot='mid', headlength=0, headaxislength=0,
                      units='inches',
                      scale=2.54,  # 2.54 for 1 cm : 100 %
                      scale_units='inches',
                      width=0.03, cmap='Accent')
    ax.coastlines()
    fig.tight_layout()
    fig.colorbar(aniso, spacing='uniform')
    plt.show()


ans_2v['Cluster'] = 0
# plot_quiver(ans_2v)


def reorient(x):
    # reorients a vector from 0-360 to -90 to 90 degrees
    val = x % 360
    if 90 < val < 270:
        return val - 180
    if val >= 270:
        return val - 360
    else:
        return val


ans_2v.Azimuth = ans_2v.Azimuth.apply(reorient)

kmeans = KMeans(n_clusters=3)
y = kmeans.fit_predict(ans_2v[['Longitude', 'Latitude', 'Azimuth']], sample_weight=ans_2v.Strength)
# dbscan = DBSCAN(eps=5, min_samples=18)
# dbscan.fit(ans_2v[['Longitude', 'Latitude', 'Azimuth']], sample_weight=ans_2v.Strength**1)
# y = dbscan.labels_
def plot_nn():
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(ans_2v[['Longitude', 'Latitude', 'Azimuth']])
    distances, indices = nbrs.kneighbors(ans_2v[['Longitude', 'Latitude', 'Azimuth']])
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()


ans_2v['Cluster'] = y
print(ans_2v.Cluster.describe())
plot_quiver(ans_2v[ans_2v.Strength > 0.1])
