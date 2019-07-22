import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob

import os


def smooth(csv_path, weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Value'],
                       dtype={'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Value': smoothed})
    save.to_csv(csv_path + '_smooth')


def my_smooth(csv_path, weight=0.99):
    with open(csv_path, newline='') as f:
        dir = '/'.join(csv_path.split('/')[:-1])
        file_name = csv_path.split('/')[-1]
        new_file_name = 'smooth_' + file_name

        reader = csv.reader(f)
        points = [float(row[0]) for row in reader]
        smoothed_points = []
        last = points[0]
        for point in points:
            smoothed_point = last * weight + (1 - weight) * point
            smoothed_points.append(smoothed_point)
            last = smoothed_point

        with open(os.path.join(dir, new_file_name).replace('\\', '/'), 'w', newline='') as f:
            writer = csv.writer(f)
            for point in smoothed_points:
                writer.writerow((point,))


def plot():
    path_list = glob.glob('../result_EVAL/*')
    fig, axs = plt.subplots(len(path_list), 1)
    for ax, path in zip(axs, path_list):
        file_name = path.split('/')[-1]

        rewards = []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                rewards.append(float(row[0]))

        ax.plot(range(len(rewards)), rewards)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(file_name)

    plt.show()


if __name__ == '__main__':
    # plot()
    path_list = glob.glob('../result_EVAL/reward_*')
    for path in path_list:
        my_smooth(path)

    plot()
