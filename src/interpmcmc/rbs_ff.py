import numpy as np
from scipy.interpolate import RectBivariateSpline
from interpmcmc.ff import ForceField


class InterpForceField(ForceField):
    def __init__(self, rbs):
        super().__init__()
        self.rbs = rbs

    @classmethod
    def from_npy_pmf(cls, pmf, domain, pad_width=3, clip_val=20, mode='wrap'):
        rbs = build_rbs(pmf, domain, pad_width, clip_val, mode)
        return cls(rbs)

    def U(self, config):
        x = config[..., 0]
        y = config[..., 1]
        return self.rbs(x, y, grid=False)

    def F(self, config):
        x = config[..., 0]
        y = config[..., 1]
        f_x = -1 * self.rbs(x, y, grid=False, dx=1)
        f_y = -1 * self.rbs(x, y, grid=False, dy=1)
        return np.stack([f_x, f_y], axis=-1)


def build_padded_ax(dmn, npnts, pad_width):
    dx = (dmn[1] - dmn[0]) / npnts
    npnts_padded = npnts + 1 + 2 * pad_width
    bottom = dmn[0] - pad_width * dx
    top = dmn[1] + pad_width * dx
    padded_edges = np.linspace(bottom, top, npnts_padded)
    padded_ax = (padded_edges[1:] + padded_edges[:-1]) / 2.
    return padded_ax


def pad_pmf(pmf, domain, pad_width=3, clip_val=20, mode='wrap'):
    dmn_range_x = domain[0]
    dmn_range_y = domain[1]
    npnts_x, npnts_y = pmf.shape

    # Build xax values for padded pmf
    xax_padded = build_padded_ax(dmn_range_x, npnts_x, pad_width)
    yax_padded = build_padded_ax(dmn_range_y, npnts_y, pad_width)

    # Avoid inf values in pmf
    if clip_val is not None:
        pmf = np.clip(pmf, None, clip_val)

    # PMF w. Padded PBCs
    pd = ((pad_width, pad_width), (pad_width, pad_width))
    padded_pmf = np.pad(pmf, pd, mode=mode)

    return padded_pmf, xax_padded, yax_padded


def build_rbs(pmf, domain, pad_width=3, clip_val=20., mode='wrap'):
    pmf, xax, yax = pad_pmf(pmf, domain, pad_width, clip_val=20., mode=mode)

    rbs = RectBivariateSpline(xax, yax, pmf)
    return rbs
