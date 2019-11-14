import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

import sys

args = sys.argv

root_path = args[1]


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
        ax.plot(data.iloc[:, 0], data.iloc[:, 1],
                color=np.random.rand(3), label=os.path.basename(path))

    ax.legend(loc='best')

    plt.savefig(os.path.join(root_path, 'result.svg'))
    plt.show()
