from bluepy.btle import Scanner
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

C = Calibrator(d0, rss_d0, gamma)
particleFilter = ParticleFilter(250, roomWidth, roomLength)
particle, w = particleFilter.generate_initial_belief(roomWidth, roomLength)
predictionPlotter = PredictionPlotter(roomWidth, roomLength)

time_slot = 0
while True:
    try:
        ble_list = Scanner().scan(1.0)
        for dev in ble_list:
            for i in range(len(beaconList)):
                if str(dev.addr).startswith(macList[i]):
                    rssMeasured[i][time_slot] = float(dev.rssi)
        
        rss = np.asarray([np.mean(rssMeasured[i][:time_slot]) for i in range(6)])
        #print(rss)
        d = C.get_distance(rss)
        print(d)
        dEff1 = d[~np.isnan(d)]
        #dEff1[dEff1 > 10] = 10
        dEff = dEff1
        #dEff = dEff1[dEff1 < 10]
        beaconLocationEff = beaconLocation[~np.isnan(d)]
        #beaconLocationEff = beaconLocationEff[dEff1 < 10]
        particleFilter.update_weight(dEff, beaconLocationEff, 3)
        particle, w = particleFilter.resample(0.1)
        predictedLocation = particleFilter.predict_location()
        predictionPlotter.plot_update(beaconLocationEff, particle, predictedLocation, dEff)
        time_slot = time_slot + 1

    except:
        raise Exception("Error occurred")