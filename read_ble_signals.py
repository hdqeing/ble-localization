"""
Plots Received Signal Strength (RSS) over time.

run using "sudo -E python3 <this-files-name>.py" (sudo for bluetooth usage. -E to keep your environment in order to use
installed packages)

Enter data into mac_to_data, each entry consists of:
    uid: (mac_address, [])
where uid is the descriptor of the BT device, mac_address is the (unique prefix of the) mac address of the device, and []
is the empty array where RSS values are stored.
"""
from email import header
from bluepy.btle import Scanner
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = 0.59

y = 2.62

filename = "x_"+str(x)+"y_"+str(y)+".csv"

time = 120

# dict: int -> tuple - tuple is str, [] where str is lower case string of mac address (start)
mac_to_data = {
    "A51f": ("f2:f4:8b", []),
    "WCc5": ("d9:2b:31", []),
    "Na0q": ("f3:46:7a", []),
    "B-vdxBwg": ("a0:e6:f8:77:ad:53", []), 
    "B-JCd2IY": ("a0:e6:f8:77:ac:0b", []),
    "B-6FSrDg": ("24:71:89:17:58:1d", []),
}

beaconList = list(mac_to_data.keys())
macList = [mac_to_data[b][0] for b in beaconList]

data = np.zeros((6, time))

ite = 0
while ite < time:
    try:
        ble_list = Scanner().scan(1.0)
        for dev in ble_list:
            for i in range(len(beaconList)):
                if str(dev.addr).startswith(macList[i]):
                    data[i][ite] = float(dev.rssi)
        ite = ite + 1
    except:
        raise Exception("Error occurred")
df = pd.DataFrame(data)
df.to_csv(filename, header=None)

    



'''
while True:
    try:
        ble_list = Scanner().scan(1.0)
        for dev in ble_list:
            for key in mac_to_data.keys():
                if str(dev.addr).startswith(mac_to_data[key][0]):
                    mac_to_data[key][1].append(float(dev.rssi))
        plt.clf()
        for key in mac_to_data.keys():
            plt.plot(mac_to_data[key][1], label=key)
        plt.ylabel("RSS / db")
        plt.legend()
        plt.pause(0.05)
        plt.draw()
    except:
        d = {}
        for key in mac_to_data.keys():
            d[key] = mac_to_data[key][1]
        
        #df = pd.DataFrame(data=d)
        df = pd.DataFrame(data = dict([ (k,pd.Series(v[1])) for k,v in mac_to_data.items() ]))
        df.to_csv (r'./0.5mNewData.csv', index = False)
        raise Exception("Error occurred")
'''

