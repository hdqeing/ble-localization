import numpy as np

class Calibrator:

    def __init__(self, d0, rss_d0, gamma):
        self.d0 = d0
        self.rss_d0 = rss_d0
        self.gamma = gamma

    def get_distance(self, rss):
        return self.d0 * np.power(10, (self.rss_d0 - rss) / (10 * self.gamma))