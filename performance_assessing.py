import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from Calibrator import Calibrator
from ParticleFilter import ParticleFilter 
from PredictionPlotter import PredictionPlotter

roomWidth = 6.81
roomLength = 8.77
windowSize = 10
rssMeasured = np.empty((6, 999))
rssMeasured[:] = np.nan

#beacon name to mac address
beaconMac = {
    "A51f": "f2:f4:8b", 
    "WCc5": "d9:2b:31", 
    "Na0q": "f3:46:7a", 
    "B-vdxBwg": "a0:e6:f8:77:ad:53", 
    "B-JCd2IY": "a0:e6:f8:77:ac:0b", 
    "B-6FSrDg": "24:71:89:17:58:1d",
}

beaconList = list(beaconMac.keys())
macList = list(beaconMac.values())

#beacon property
beaconLocation = np.asarray([[2.09,5.92], [0,roomLength], [roomWidth,0], [4.44, 5.91], [2.17,3.11], [4.41, 3.11]])

params = pd.read_csv("params.csv")
d0 = params.iloc[:,0].to_numpy()
rss_d0 = params.iloc[:,1].to_numpy()
gamma = params.iloc[:,2].to_numpy()

samplePos = [[0.54, 3.66], [0.59, 2.62], [1.32, 8.26], [2.67, 4.19], [2.78, 0.7], [2.97, 7.42], [5.10, 1.69], [5.97, 7.91], [6.07, 4.81]]

bestPos = [[5.97, 7.91]]



for s in bestPos:
    C = Calibrator(d0, rss_d0, gamma)
    particleFilter = ParticleFilter(500, roomWidth, roomLength)
    particle, w = particleFilter.generate_initial_belief(roomWidth, roomLength)
    predictionPlotter = PredictionPlotter(roomWidth, roomLength)
    df = pd.read_csv("performance_data/x_"+ f"{s[0]:.2f}" +"y_" + f"{s[1]:.2f}" +".csv", header = None)
    data = df.to_numpy()
    time_slot = 0
    error = []
    while time_slot < len(data[0]):
        rss = np.asarray([np.mean(data[i][:time_slot]) for i in range(6)])
        d = C.get_distance(rss)
        #print(d)
        d[1] = np.nan
        d[2] = np.nan
        dEff1 = d[~np.isnan(d)]
        #dEff1[dEff1 > 10] = 10
        dEff = dEff1
        #dEff = dEff1[dEff1 < 10]
        beaconLocationEff = beaconLocation[~np.isnan(d)]
        #beaconLocationEff = beaconLocationEff[dEff1 < 10]
        particleFilter.update_weight(dEff, beaconLocationEff, 3)
        particle, w = particleFilter.resample(0.1)
        predictedLocation = particleFilter.predict_location()
        predictionPlotter.plot_update(beaconLocationEff, particle, predictedLocation, dEff, s)
        #if time_slot == 0 or time_slot == 1 or time_slot == 32:
        #    predictionPlotter.save_plot(time_slot)
        error.append(particleFilter.calc_error(s))
        time_slot = time_slot + 1
    

    """
    plt.figure(figsize=(23.5/2.54, 20/2.54))
    plt.plot(error)
    plt.xlabel("time/s")
    plt.ylabel("error/m")
    plt.tight_layout()
    plt.savefig("particleFilterError.png")"""
