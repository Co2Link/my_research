import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

import sys

COLOR = ['b','g','r','c','m','y','k','w']


def plot_multiple(root_path):
    """Plot multiple line

    each row should be like (index,value),
    and line are labed by its file name

    Args:
        root_path: path that csv files are in
    """
    paths = glob.glob(root_path+'/*.csv')

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    for i, path in enumerate(sorted(paths)):
        data = pd.read_csv(path, header=None)
        
        color = path.split('-')[1][0] if path.split('-')[1][0] in COLOR else np.random.rand(3)
        ax.plot(data.iloc[:, 0], data.iloc[:, 1],
                color=color, label=os.path.basename(path))

    ax.legend(loc='best')

    plt.savefig(os.path.join(root_path, 'result.svg'))
    plt.show()

def plot_multiple_avg(root_path,save_file_name = 'result',num_iter = 1000000):
    """
    Avgs:
        root_path: path that cotain directories
        num_iter: the iteration num of csv file
    """
    fig,ax = plt.subplots()
    fig.set_size_inches(12,8)

    for i,dir_path in enumerate(glob.glob(root_path+'/*')):
        print(dir_path)
        none = []
        for csv_file in glob.glob(dir_path+'/*.csv'):
            values = pd.read_csv(csv_file).iloc[:,1].values
            values = np.append(values,[values[-1] for _ in range(num_iter-len(values))])
            none.append(values)
        indexes = list(range(num_iter))
        none = np.array(none)
        none_mean = np.mean(none,axis=0)
        none_std = np.std(none,axis=0)
        ax.fill_between(indexes,none_mean+none_std,none_mean-none_std,facecolor=COLOR[i],alpha=0.5)

        ax.plot(indexes,none_mean,color=COLOR[i],label=os.path.basename(dir_path))
    
    ax.legend(loc='best')

    plt.savefig('{}.svg'.format(save_file_name))
    plt.show()

def main():
    args = sys.argv
    root_path = args[1]

    # plot_multiple(root_path)
    plot_multiple_avg(root_path)

if __name__ == "__main__":
    main()