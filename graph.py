import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import argparse

import sys

COLOR = ['b','g','r','c','m','y','k','w']


def plot_multiple(root_path,header=False,save_file_name = 'result'):
    """Plot multiple line

    each row should be like (index,value),
    and line are labed by its file name

    Args:
        root_path: path that csv files are in
    """
    header = 'infer' if header else None
    
    paths = glob.glob(root_path+'/*.csv')

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    for path in sorted(paths):
        data = pd.read_csv(path, header=header)
        
        color = path.split('-')[1][0] if '-' in path and path.split('-')[1][0] in COLOR else np.random.rand(3)
        ax.plot(list(range(len(data))), data.iloc[:, -1],
                color=color, label=os.path.basename(path))

    ax.legend(loc='best')

    plt.savefig(save_file_name)
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

    plot_multiple(root_path)
    # plot_multiple_avg(root_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--header',action='store_true')
    parser.add_argument('-p','--path',type=str,default='')
    parser.add_argument('--plot_avg',action='store_true')

    args = parser.parse_args()

    HEADER = args.header
    PATH = args.path
    PLOT_AVG = args.plot_avg

    if PLOT_AVG:
        plot_multiple_avg(PATH)
    else:
        plot_multiple(PATH,header=HEADER)