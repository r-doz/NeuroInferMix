class GaussianMix():

    """Representation of Univariate Gaussian Mixture"""

    def __init__(self, pi, mu, sigma):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def n_comp(self):
        return self.pi.shape[0]
    
    def n_dim(self):
        return self.mu.shape[1]
    
    def __repr__(self):
        str_repr = 'pi: ' + str(self.pi) + '\nmu: ' + str(self.mu) + '\nsigma: ' + str(self.sigma)
        return str_repr   