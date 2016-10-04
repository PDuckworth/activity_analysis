#!/usr/bin/env python
__author__ = 'p_duckworth'
import sys, os
import numpy as np
import subprocess
import matplotlib.pylab as plt
import cPickle as pickle


def save_event(e, loc=None):
    """Save the event into an Events folder"""

    p = e.dir.split('/')
    if loc != None:
        p[4] = loc
    new_path = '/'.join(p[:-1])

    if not os.path.isdir(new_path):
        os.system('mkdir -p ' + new_path)
    f = open(os.path.join(new_path, p[-1] +".p"), "w")
    pickle.dump(e, f, 2)
    f.close()

def load_e(directory, event_file):
    """Loads an event file along with exception raise msgs"""

    try:
        file = directory + "/" + event_file
        with open(file, 'r') as f:
            e = pickle.load(f)
        return e

    except (EOFError, ValueError, TypeError), error:
        print "Load Error: ", error, directory, event_file
        return None


def load_all_learning_files(accu_path):
    # print "loading files..."

    with open(accu_path + "/code_book.p", 'r') as f:
        code_book = pickle.load(f)

    with open(accu_path + "/graphlets.p", 'r') as f:
        graphlets = pickle.load(f)

    with open(accu_path + "/feature_space.p", 'r') as f:
        data = pickle.load(f)

    return code_book, graphlets, data


def genome(filepath, data1, yrange, title="", vis=False):
    t = np.arange(0.0, len(data1), 1)
    #print "data", min(data1), max(data1), sum(data1)/float(len(data1))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlim([-1,65])
    ax1.vlines(t, [0], data1)
    ax1.set_xlabel('code words', fontsize=20)
    ax1.set_ylim(yrange)

    ax1.set_title(title, fontsize=25)
    ax1.grid(True)
    filepath_g = os.path.join(filepath, "graphs")
    if not os.path.exists(filepath_g):
        os.makedirs(filepath_g)

    filename = title.replace(" ", "_")+".png"
    fig.savefig(filepath_g + "/" + filename, bbox_inches='tight')
    if vis:
        plt.show()



def screeplot(filepath, sigma, comps, div=2, vis=False):
    y = sigma
    x = np.arange(len(y)) + 1

    plt.subplot(2, 1, 1)
    plt.plot(x, y, "o-", ms=2)

    xticks = ["Comp." + str(i) if i%2 == 0 else "" for i in x]

    plt.xticks(x, xticks, rotation=45, fontsize=20)

    # plt.yticks([0, .25, .5, .75, 1], fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("Variance", fontsize=20)
    plt.xlim([0, len(y)])
    plt.title("Plot of the variance of each Singular component", fontsize=25)
    plt.axvspan(10, 11, facecolor='g', alpha=0.5)

    filepath_g = os.path.join(filepath, "graphs")
    if not os.path.exists(filepath_g):
        os.makedirs(filepath_g)

    plt.savefig(filepath_g + "/scree_plot.png", bbox_inches='tight')
    if vis:
        plt.show()
