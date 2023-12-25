import matplotlib.pyplot as plt
import csv
import numpy as np
import os

def filter_dirs(path=None):
    if path is None:
        mypath = os.getcwd()
        dirs = [dirname for dirname in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, dirname))]
    else:
        mypath = path
        dirs = [path + '/' + dirname for dirname in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, dirname))]

    dirs_to_filter_out = ['.idea','__pycache__','data']
    filtered_dirs = []
    for dirname in dirs:
        if dirname not in dirs_to_filter_out:
            filtered_dirs.append(dirname)
    return filtered_dirs


dirs = filter_dirs('result_colab')

out = []
for dir in dirs:
    file = dir + '/test_results.csv'
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        p = []
        s = []
        for row in spamreader:
            p.append(int(row[0]))
            s.append(float(row[2]))
        out.append((p, s))

out = np.array(out)
for idx, o in enumerate(out):
    x = o[0]
    y = o[1]
    plt.plot(x, y)
    name = dirs[idx]
plt.legend(dirs)
plt.show()
