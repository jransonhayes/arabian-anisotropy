import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, halfnorm

PERIODS = [5, 8, 10, 18, 25, 36, 46, 55, 70]


def read_raw_file(period):
    print(f"Loading raypath file {period}s to pandas")
    df = pd.read_csv(f"data/raypaths_{period}s.txt", delim_whitespace=True)
    return df


def generate_station_file():
    stations = np.array([])
    latitudes = np.array([])
    longitudes = np.array([])

    for period in PERIODS:  # this is not very efficient...
        raypaths = read_raw_file(period)
        sta1 = raypaths.sta1.to_numpy()
        sta2 = raypaths.sta2.to_numpy()
        lat_sta1 = raypaths.lat_sta1.to_numpy()
        lon_sta1 = raypaths.lon_sta1.to_numpy()
        lat_sta2 = raypaths.lat_sta2.to_numpy()
        lon_sta2 = raypaths.lon_sta2.to_numpy()

        stations = np.append(stations, np.append(sta1, sta2))
        latitudes = np.append(latitudes, np.append(lat_sta1, lat_sta2))
        longitudes = np.append(longitudes, np.append(lon_sta1, lon_sta2))

    data = np.array([stations, latitudes, longitudes])
    # data = np.sort(data, axis=1)

    data = pd.DataFrame(data.T)
    data[1] = pd.to_numeric(data[1])
    data[2] = pd.to_numeric(data[2])
    data.sort_values(0, inplace=True)
    data.drop_duplicates(subset=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data.columns = ['Station', 'Latitude', 'Longitude']
    data.to_csv('data/stations.csv')


def generate_tomo_file(period, station_nums):
    df = read_raw_file(period)
    output = ""

    for row in df.itertuples():
        seg = f"z{row[0] + 1} path_{station_nums[row[7]]}_{station_nums[row[8]]} \n"
        seg += f"{row[1]} {row[2]} {row[3]} {row[4]} \n"
        seg += f"{row[5]} \n"
        seg += f"{row[6]} \n"
        output += seg

    with open(f'clean/intomo_{period}.rayl', 'w') as f:
        print(f"Printing to 'clean/intomo_{period}.rayl'")
        f.write(output)


def generate_tomo_files():
    # load stations and create a dictionary of station names to numbers
    df = pd.read_csv('data/stations.csv', index_col=0)
    stations = df.Station.to_list()
    station_nums = {k: (v + 1) for (v, k) in enumerate(stations)}

    for i in PERIODS:
        print(f"Period: {i}s")
        generate_tomo_file(i, station_nums)

    print('Done!')


def plot_hist():
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 10))
    for period in enumerate(PERIODS):
        # read data for a period
        df = read_raw_file(period[1])
        data = df.vg.to_numpy()

        coords = (period[0] // 3, period[0] % 3)
        ax = axs[coords]
        # plot histogram
        n_bins = 30
        n, bins, patches = ax.hist(data, bins=n_bins, density=1)
        # fit normal PDF
        mu, sigma = norm.fit(data)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        ax.plot(bins, y)

        # plot mean and +-2sigma
        ax.axvline(mu, color='black')
        ax.axvline(mu - 2 * sigma, color='red')
        ax.axvline(mu + 2 * sigma, color='red')

        # labels
        ax.set_title(f"{period[1]}s, {len(data)} paths; μ: {mu:.3f} σ: {sigma:.3f}")
        ax.set_xlabel("Group Velocity (km/s)")
        ax.set_ylabel("Probability Density")

    plt.suptitle(f"Pre-tomo Group Velocities. {n_bins} bins")
    plt.show()

    return


def trim_nsig(period, n=2):
    df = read_raw_file(period)
    data = df.vg.to_numpy()

    mu, sigma = norm.fit(data)

    disc = data[(data >= mu - n * sigma) | (data <= mu + n * sigma)]
    return disc


def main():
    # plot_hist()
    # post_tomo_hist()
    # trimmed = trim_so(0.4)
    #
    df = pd.read_csv('data/stations.csv', index_col=0)
    stations = df.Station.to_list()
    station_nums = {k: (v + 1) for (v, k) in enumerate(stations)}
    generate_tomo_file(8, station_nums, False)

    # generate_station_file()
    # generate_tomo_files()
    print()


if __name__ == '__main__':
    main()
