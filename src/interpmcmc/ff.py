import numpy as np
from interpmcmc.utils import min_image


class ForceField(object):
    def U(self, config):
        raise NotImplementedError

    def F(self, config):
        raise NotImplementedError

    def __call__(self, config):
        return self.F(config)


class HarmonicPotential(ForceField):
    def __init__(self, fk, center, domain=None):
        if domain is not None:
            self.dmn_width = domain[:, 1] - domain[:, 0]
        else:
            self.dmn_width = None

        self.fk = fk
        self.center = center

    def U(self, config):
        distance_vec = config - self.center
        if self.dmn_width is not None:
            distance_vec = min_image(distance_vec, self.dmn_width)

        U_components = 0.5 * self.fk * (distance_vec**2)
        return np.sum(U_components, axis=-1)

    def F(self, config):
        distance_vec = config - self.center
        if self.dmn_width is not None:
            distance_vec = min_image(distance_vec, self.dmn_width)

        return -1 * self.fk * distance_vec


class Compose(ForceField):
    def __init__(self, ff_list):
        self.ff_list = ff_list

    def U(self, config):
        return np.sum([ff.U(config) for ff in self.ff_list], axis=0)

    def F(self, config):
        # f_list = [ff.F(config) for ff in self.ff_list]
        return np.sum([ff.F(config) for ff in self.ff_list], axis=0)
