from sqlite3 import apilevel
from sre_constants import GROUPREF_EXISTS
from tkinter import N
from turtle import distance
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset



beaconNames = ["A51f", "WCc5", "Na0q", "B-vdxBwg", "B-JCd2IY", "B-6FSrDg"]

beaconLocation = []


filenames = ["0.1mNewData.csv", "0.5mNewData.csv", "1.0mNewData.csv", "1.5mNewData.csv", "2.0mNewData.csv", "2.5mNewData.csv", 
             "3.0mNewData.csv", "3.5mNewData.csv", "4.0mNewData.csv", "4.5mNewData.csv", "5.0mNewData.csv", "5.5mNewData.csv", 
             "6.0mNewData.csv", "7.0mNewData.csv", "8.0mNewData.csv", "9.0mNewData.csv", "10mNewData.csv"]

'''
filenames = ["0.5m.csv", "1m.csv", "1.5m.csv", "2.5m.csv", "3m.csv", 
             "3.5m.csv", "4m.csv", "4.5m.csv", "5m.csv", "5.5m.csv", "6m.csv",
             "7m.csv", "8m.csv", "9m.csv", "10m.csv"]
'''


#dist = np.asarray([float(i[:-1]) for i in distList])


distList = [f[:-11] for f in filenames]

dist = np.asarray([float(i[:-1]) for i in distList])

filenames = ["used_data/"+filename for filename in filenames]

def preprocess():
    #sum up data for each beacon
    for beacon in beaconNames:
        rss = {}
        for i in range(len(filenames)):
            df = pd.read_csv(filenames[i])     
            data = df.loc[:,beacon].to_numpy()
            data = data.flatten('F')
            filtered_data = data[~np.isnan(data)]
            rss[filenames[i][10:-11]] = filtered_data
        dfBeacon = pd.DataFrame(data = dict([ (k,pd.Series(v)) for k,v in rss.items() ]))
        dfBeacon.to_csv (r'./'+beacon+'.csv', index = False)

def plot_hist(beacon, n_row, n_column):
    #load in data
    df = pd.read_csv(beacon + '.csv')
    #distList = list(df.columns.values)
    #plot histogram
    fig, ax = plt.subplots(n_row, n_column)
    for i in range(len(distList)):
        ax[i//n_column, i % n_column].hist(df.loc[:,distList[i]],bins = 30)
        ax[i//n_column, i % n_column].title.set_text(distList[i])
    plt.tight_layout()
    plt.show()

def plot_boxplot(beacon, fig_width = 8, fig_height = 6):
    #load in data
    df = pd.read_csv(beacon + '.csv')
    #distList = list(df.columns.values)
    #plot boxplot
    plt.figure(figsize=(fig_width, fig_height), dpi=80)
    rss =[]
    for d in distList:
        data = df.loc[:,d]
        rss.append(data[~np.isnan(data)])
    plt.boxplot(rss, positions=dist)
    plt.title(beacon)
    plt.show()

def find_reference(beacon, plot = False):
    coeff, err = fit_model(beacon)
    iMin = np.argmin(err)
    #print("Minum at "+str(dist[iMin])+" meters.")
    if plot:
        plt.plot(dist, err)
        plt.title(beacon)
        plt.show()
    return coeff[iMin], dist[iMin]


def split_data(beacon, r = 0.75, rSeed = 2):
    np.random.seed(rSeed)
    df = pd.read_csv(beacon + '.csv')
    rss_training = []
    rss_testing = []
    for d in distList:
        data = df.loc[:,d]
        rss = data[~np.isnan(data)]
        idx_training = np.random.choice(len(rss), int(len(rss) * r), replace=False)
        print(len(idx_training))
        idx_all = list(range(0, len(rss)))
        rss_training.append(rss[idx_training])
        rss_testing.append(rss[[i for i in idx_all if i not in idx_training]])
    return rss_training, rss_testing

def fit_model(beacon, n_rows = 4, n_cols = 4, plot = False, r = 0.75, rSeed = 2):
    """
    Plot fitting performance against different reference (d_0) 
    More Information: https://www.scirp.org/journal/paperinformation.aspx?paperid=91056
    """
    rss = np.zeros(len(dist))
    coeff = np.zeros(len(dist))
    err = np.zeros(len(dist))
    rss_training, _ = split_data(beacon, r, rSeed)
    for i in range(len(rss_training)):
        rss[i] = np.mean(rss_training[i])
    for i in range(len(dist)):
        x = np.log10(dist/dist[i])
        y = rss - rss[i]
        x = x[:,np.newaxis]
        coeff[i], err[i], _, _ = np.linalg.lstsq(x, y, rcond=None)
    if plot:
        fig, ax = plt.subplots(n_rows, n_cols)
        #plot prediction
        for i in range(len(dist)):
            ax[i//n_rows, i%n_cols].scatter(dist, rss)
            d_pred = np.linspace(0.01, 12, 100)
            y_pred = rss[i] + coeff[i] * np.log10(d_pred/dist[i])
            ax[i//n_rows, i%n_cols].plot(d_pred, y_pred)
            ax[i//n_rows, i % n_cols].title.set_text("Reference: " +distList[i])
        plt.tight_layout()
        plt.show()
    return coeff, err

def calc_d(rss, d0, rss_d0, gamma):
    return d0 * 10 ** ((rss_d0 - rss)/(10 * gamma))

def verify_model(beacon, r = 0.75, rSeed = 2, plot = False, n_rows = 5, n_cols = 5, err_allowed = 0.5):
    coeff, d0 = find_reference(beacon)
    rss_training, rss_testing = split_data(beacon, r, rSeed)
    idx = np.where(dist == d0)[0][0]
    rss_d0 = np.mean(rss_training[idx])
    acc = np.zeros(len(dist))
    for i in range(len(dist)):
        x = np.arange(len(rss_testing[i]))
        d = calc_d(np.asarray(rss_testing[i]), d0, rss_d0, coeff / -10.0)
        effPred = [pred for pred in d if (pred > dist[i] * (1-err_allowed) and pred < dist[i] * (1+err_allowed))]
        acc[i] = len(effPred)/len(d)

    if plot:
        fig, ax = plt.subplots(n_rows, n_cols)
        #plot prediction
        for i in range(len(dist)):
            x = np.arange(len(rss_testing[i]))
            d = calc_d(np.asarray(rss_testing[i]), d0, rss_d0, coeff / -10.0)
            ax[i//n_rows, i%n_cols].scatter(x, d)
            ax[i//n_rows, i%n_cols].axhline(dist[i])
            ax[i//n_rows, i%n_cols].axhline(dist[i] * (1-err_allowed), c = "r")
            ax[i//n_rows, i%n_cols].axhline(dist[i] * (1+err_allowed), c = "r")
            ax[i//n_rows, i % n_cols].title.set_text("Test: " +distList[i])          
        plt.tight_layout()
        plt.show()
    return acc

def check_fitting_performance(beacon, plot = False, windowSize = 1):
    df = pd.read_csv(beacon + '.csv')
    a, d0 = find_reference(beacon)
    idx = np.where(dist == d0)[0][0]
    rss_d0 = np.mean(df.loc[:,distList[idx]])
    deviation = np.zeros(len(dist))
    for i in range(len(distList)):
        data = df.loc[:,distList[i]]
        data = data[~np.isnan(data)]
        data_smoothed = smooth_curve(data, windowSize)
        distPred = calc_d(data_smoothed, d0, rss_d0, a / -10.0)
        distPred[distPred > 11] = np.nan
        #deviation[i] = np.mean(np.abs(distPred[~np.isnan(distPred)] - dist[i])/dist[i])
        deviation[i] = np.mean(np.abs(distPred[~np.isnan(distPred)] - dist[i]))
        print(beacon + ":" + "distance: " + distList[i] + "\t" + str(deviation[i]))
    if plot:
        plt.plot(dist, deviation, 'o')
        plt.show()
    return deviation

def smooth_curve(y, windowSize):
    y_smoothed = np.zeros(len(y)-windowSize+1)
    for i in range(len(y_smoothed)):
        y_smoothed[i] = np.mean(y[i:i+windowSize])
    return y_smoothed

def plot_original_data(beacon):
    df = pd.read_csv(beacon + '.csv')
    nData = []
    for d in distList:
        data = df.loc[:,d]
        rss = data[~np.isnan(data)]
        nData.append(len(rss))
    nMin = min(nData)
    dataBeacon = np.zeros((len(dist), nMin))
    for i in range(len(distList)):
        data = df.loc[:,distList[i]]
        rss = data[~np.isnan(data)]
        dataBeacon[i] = rss[:nMin]
    x = np.ones(dataBeacon.shape)
    for i in range(len(dist)):
        x[i] = dist[i] * x[i]
    plt.plot(x.flatten("F") + 0.01*np.random.random_sample(len(x) * len(x[0])),dataBeacon.flatten("F"), "o")
    plt.show()

def plot_example():
    df = pd.read_csv('B-vdxBwg.csv')
    fig, axs = plt.subplots(2, 1)
    axins = zoomed_inset_axes(axs[0],2,loc='lower right')
    for i in range(1,len(distList)):
        if i % 3 == 0:
            data = df.loc[:,distList[i]]
            rss = data[~np.isnan(data)]
            axs[0].plot(rss, label = distList[i])
            axins.plot(rss)
    axins.set_xlim(300,400)
    axins.set_ylim(-60,-50)
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)
    mark_inset(axs[0],axins,loc1=1,loc2=3)
    axs[0].set_title("B-vdxBwg")
    axs[0].set_xlabel("t/s")
    axs[0].set_ylabel("rssi")
    axs[0].legend(loc = 'lower center')

    mu = np.zeros(4)
    sigma = np.zeros(4)
    for i in range(1,len(distList) - 4):
        if i % 3 == 0:
            data = df.loc[:,distList[i]]
            rss = data[~np.isnan(data)]
            axs[1].hist(rss, bins = 10, label = distList[i])
            mu[i//3-1] = np.mean(rss)
            sigma[i//3-1] = np.std(rss) 
    '''
    x = np.linspace(-68, -29, 600)
    for i in range(len(mu)):
        p = scipy.stats.norm(mu[i], sigma[i]).pdf(x) * 600
        axs[1].plot(x, p)
    '''
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("original_rss.svg")


def calc_p_from_dist(x, y, d_measured, variance = 1, plot = False):
    p = np.zeros(len(d_measured))
    for i in range(len(d_measured)):
        d_real = np.linalg.norm(np.array([x,y]) - np.array(beaconLocation[i]))
        p[i] = scipy.stats.norm(d_measured[i], variance).pdf(d_real)
    return np.prod(p)

def plot_heat_map(d_measured):
    x = np.arange(-0.05, 6, 0.5)
    y = np.arange(-0.05, 8, 0.5)
    Z = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j][i] = calc_p_from_dist(x[i], y[j], d_measured)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, Z)
    ax.scatter(np.array(beaconLocation)[:,0], np.array(beaconLocation)[:,1])
    fig.set_figheight(8)
    fig.set_figwidth(6)
    plt.show()

"""
params = np.zeros((6, 3))
for i in range(len(beaconNames)):
    data_training, data_testing = split_data(beaconNames[i])
    a, d0 = find_reference(beaconNames[i])
    gamma = a / -10.0
    idx = np.where(dist == d0)[0][0]
    rss_d0 = np.mean(data_training[idx])
    params[i][0] = d0
    params[i][1] = rss_d0
    params[i][2] = gamma
paramsDf = pd.DataFrame(params)
paramsDf.to_csv(r"./params.csv", index=False)
#print(b + "\nd0: " + str(d0) + "\nrss_d0: " + str(rss_d0) + "\ngamma: " + str(gamma))
"""

for i in [3]:
    plot_example()