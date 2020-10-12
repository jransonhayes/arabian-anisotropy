import numpy as np
import pandas as pd

PERIODS = [5, 8, 10, 18, 25, 36, 46, 55, 70]


def read_raw_file(period):
    print("Loading raypath file to pandas")
    df = pd.read_csv(f"data/raypaths_{period}s.txt", delim_whitespace=True)
    return df


def generate_station_file():
    period = 5

    raypaths = read_raw_file(period)
    sta1 = raypaths.sta1.to_numpy()
    sta2 = raypaths.sta2.to_numpy()
    lat_sta1 = raypaths.lat_sta1.to_numpy()
    lon_sta1 = raypaths.lon_sta1.to_numpy()
    lat_sta2 = raypaths.lat_sta2.to_numpy()
    lon_sta2 = raypaths.lon_sta2.to_numpy()

    stations = np.append(sta1, sta2)
    latitudes = np.append(lat_sta1, lat_sta2)
    longitudes = np.append(lon_sta1, lon_sta2)
    data = np.array([stations, latitudes, longitudes])
    data = np.sort(data, axis=1)

    sta_keys = []
    cleaned_data = []
    for col in np.transpose(data):
        if col[0] not in sta_keys:
            cleaned_data.append(list(col))
            sta_keys.append(col[0])
    df = pd.DataFrame(cleaned_data)
    df.columns = ['Station', 'Latitude', 'Longitude']
    df.to_csv('data/stations.csv')


def generate_tomo_file():
    # load stations and create a dictionary of station names to numbers
    df = pd.read_csv('data/stations.csv', index_col=0)
    stations = df.Station.to_list()
    station_nums = {k: (v + 1) for (v, k) in enumerate(stations)}

    for i in PERIODS:
        print(f"Period: {i}s")
        df = read_raw_file(i)
        output = ""
        for row in df.itertuples():
            seg = f"z{row[0]+1} path_{station_nums[row[7]]}_{station_nums[row[8]]} \n"
            seg += f"{row[1]} {row[2]} {row[3]} {row[4]} \n"
            seg += f"{row[5]} \n"
            seg += f"{row[6]} \n"
            output += seg

        with open(f'clean/intomo_{i}.rayl', 'w') as f:
            print(f"Printing to 'clean/intomo_{i}.rayl'")
            f.write(output)

    print('Done!')


def main():
    generate_station_file()
    generate_tomo_file()


if __name__ == '__main__':
    main()
