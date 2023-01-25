import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class PredictionPlotter():

    def __init__(self, width, length):
        self.fig, self.ax = plt.subplots(1,1)
        self.fig.tight_layout()
        self.width = width
        self.length = length
        self.fig.set_size_inches(width, length)

    def plot_beacons(self, beaconLocation):
        self.ax.scatter(beaconLocation[:,0], beaconLocation[:,1], c = 'b', s = 100, label = "beacon")

    def plot_particles(self, particles):
        self.ax.scatter(particles[:, 0], particles[:, 1], marker="x", c = 'r', alpha = 0.15, label = "particle")
    
    def plot_prediction(self, predictedLocation):
        self.ax.scatter(predictedLocation[0], predictedLocation[1], c = 'g', label = "prediction")

    def plot_reference_circle(self, beaconLocation, d):
        for i in range(len(d)):
            c = plt.Circle((beaconLocation[i][0], beaconLocation[i][1]), d[i], fill = False)
            self.ax.add_patch(c)

    def plot_real_loc(self, loc):
        self.ax.plot([loc[0]], [loc[1]], "s", c='black', label="true location")


    def save_plot(self, name):
        self.fig.savefig(str(name)+".png")

    def plot_update(self, beaconLocation, particles, predictedLocation, d, loc):
        plt.cla()
        self.plot_beacons(beaconLocation)
        self.plot_particles(particles)
        self.plot_prediction(predictedLocation)
        self.plot_reference_circle(beaconLocation, d)
        self.plot_real_loc(loc)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.length)
        plt.legend(loc = 2, fontsize="xx-large")
        plt.draw()
        plt.pause(0.001)
