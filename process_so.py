from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
from formatter import read_raw_file


def read_so(so=None):
    if so is None:
        so = glob('data/so/so_*')[0]

    with open(so, 'r') as f:
        so = f.read()

    data = so.split('couche):\n')[1].split('moyenne')[0]

    data = data.split()
    # the formatting in the so file is atrocious: the length of the path code subtracts from the characters used to
    # render the word 'path', so it is not possible to split simply on whitespace.
    data = [i for i in data if i[0] != 'p']

    clean_data = []
    for i in range(len(data) // 2):
        a = data[2 * i]
        b = data[2 * i + 1]
        clean_data.append([a, float(b)])

    clean_data = pd.DataFrame(clean_data).set_index(0)
    clean_data.columns = ['Error']
    return clean_data


def trim_so(level):
    df = read_so()

    select = df[df.Error <= level]
    print(f"{100 * (1 - len(select) / len(df)):.1f}% of data excluded")
    return select.index.to_list()


def post_tomo_hist(so=None):
    data = read_so(so)

    fig, ax = plt.subplots()
    nbins = 70
    n, bins, patches = ax.hist(data, bins=nbins)
    mu, sigma = halfnorm.fit(data.to_numpy())
    # x=np.linspace(0.001,.99, 100)
    # ax.plot(x,halfnorm.pdf(x, mu, sigma), linewidth=0.8, linestyle='--')
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    # ax.plot(bins, y)

    ax.set_title(f'Post-tomo Error Histogram - {nbins} bins')
    ax.set_xlabel('Error')
    ax.set_ylabel('Count')
    plt.show()


def main(period):
    so_path = f"data/so/so_{period}_75_UAE"
    rayl = f"clean/intomo_{period}.rayl"

    # plot the distribution of errors after tomographic inversion
    post_tomo_hist(so_path)
    # read so file into DataFrame
    so_data = read_so(so_path)
    # read raypaths into DataFrame
    raypaths = read_raw_file(period)

    # find raypaths to be kept after trimming
    trimmed_so = trim_so(1.0)

    # import and enumerate stations
    stations = pd.read_csv('data/stations.csv', index_col=0)
    stations = stations.Station.to_list()
    station_nums = {k: (v + 1) for (v, k) in enumerate(stations)}

    output = ""
    for row in raypaths.itertuples():
        if f"z{row[0] + 1}" in trimmed_so:
            seg = f"z{row[0] + 1} path_{station_nums[row[7]]}_{station_nums[row[8]]} \n"
            seg += f"{row[1]} {row[2]} {row[3]} {row[4]} \n"
            seg += f"{row[5]} \n"
            seg += f"{row[6]} \n"
            output += seg

    with open(f'clean_trimmed/intomo_{period}.rayl', 'w') as f:
        print(f"Printing to 'clean_trimmed/intomo_{period}.rayl'")
        f.write(output)

    return


if __name__ == "__main__":
    main(18)
