import numpy as np
import scipy.stats
import random


np.random.seed(2)
class ParticleFilter():
    
    def __init__(self, nParticles, roomWidth, roomLength):
        self.nParticles = nParticles
        self.roomWidth = roomWidth
        self.roomLength = roomLength
        

    def generate_initial_belief(self, roomWidth, roomLength):
        self.particles = np.random.uniform([0,0], [roomWidth, roomLength], (self.nParticles, 2))
        self.weight = 1 / self.nParticles * np.ones(self.nParticles)
        return self.particles, self.weight

    @staticmethod
    def calc_p_from_dist(particleLocation, beaconLocation, d_measured, variance = 1):
        x = particleLocation[0]
        y = particleLocation[1]
        p = np.zeros(len(d_measured))
        for i in range(len(d_measured)):
            d_real = np.linalg.norm(np.array([x,y]) - np.array(beaconLocation[i]))
            p[i] = scipy.stats.norm(d_measured[i], variance).pdf(d_real)
        return np.prod(p)

    def update_weight(self, d, beaconLocation, var = 1):
        for i in range(self.nParticles):
            self.weight[i] = self.calc_p_from_dist(self.particles[i], beaconLocation, d, var)
            self.weight[i] = self.weight[i] * self.check_boundary(self.particles[i])
        self.weight = self.weight / sum(self.weight)
        return self.particles, self.weight

    def resample(self, variance):
        self.particles = random.choices(self.particles, weights = self.weight, k = self.nParticles)
        offsetX = np.asarray([np.asanyarray([i, 0]) for i in np.random.normal(0, variance, self.nParticles)])
        offsetY = np.asarray([np.asanyarray([0, i]) for i in np.random.normal(0, variance, self.nParticles)])
        self.particles = self.particles + offsetX + offsetY
        self.weight = 1 / self.nParticles * np.ones(self.nParticles)
        return self.particles, self.weight

    def predict_location(self):
        x = np.sum(np.array([self.particles[i][0] * self.weight[i] for i in range(self.nParticles)]))
        y = np.sum(np.array([self.particles[i][1] * self.weight[i] for i in range(self.nParticles)]))
        self.location = [x, y]
        return self.location

    def calc_error(self, realLoc):
        return np.linalg.norm(np.array(self.location)- np.array(realLoc))

    def check_boundary(self, pos):
        x = pos[0]
        y = pos[1]

        if x < 0 or x > self.roomWidth:
            return 0
        elif y < 0 or y > self.roomLength:
            return 0
        else:
            return 1
