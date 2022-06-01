import numpy as np


def overdamped_langevin(cfg_0, force_function, periodicity_fxn=None,
                        nsteps=1E5, burnin=100,
                        dt=0.001, kT=1.):
    """
    Runs overdamped Langevin given a force function
    """
    cfg = cfg_0
    cfg_shape = cfg.shape
    R_n = np.random.normal(0, 1)
    traj = []
    sig = np.sqrt(kT * dt / 2)
    for j in range(int(nsteps + burnin)):
        # Run Overdamped BAOAB algorithm
        rando = np.random.normal(0, 1, size=cfg_shape)
        force = force_function(cfg)
        cfg += dt * force + sig * (rando + R_n)
        R_n = rando

        # Optionally enforce periodicity
        if periodicity_fxn is not None:
            cfg = periodicity_fxn(cfg)
        traj.append(np.copy(cfg))
    return np.array(traj)[burnin:]


def metropolis(cfg_0, U_fxn, periodicity_fxn=None,
               nsteps=1E5, burnin=100,
               dx=0.001, kT=1.):
    """
    Runs overdamped Langevin given a force function
    """
    cfg = cfg_0
    cfg_shape = cfg.shape
    traj = []
    U_old = U_fxn(cfg)
    for j in range(int(nsteps + burnin)):
        # Take step
        rando = np.random.normal(0, 1, size=cfg_shape) * dx
        cfg_proposed = cfg + rando

        # Optionally enforce periodicity
        if periodicity_fxn is not None:
            cfg_proposed = periodicity_fxn(cfg_proposed)
        
        # Accept / Reject
        U_new = U_fxn(cfg_proposed)
        accept_prob = np.exp(-(U_new - U_old) / kT)
        if np.random.rand() < accept_prob:
            cfg = cfg_proposed
            U_old = U_new

        traj.append(np.copy(cfg))
    return np.array(traj)[burnin:]


class PeriodicityEnforcer(object):
    def __init__(self, domain):
        super().__init__()
        domain = np.asarray(domain)
        self.domain_start = domain[:, 0]
        self.domain_end = domain[:, 1]
        self.domain_width = self.domain_end - self.domain_start

    def __call__(self, cfg):

        # print(self.domain_start)
        cfg_shifted = cfg - self.domain_start
        cfg_mod = cfg_shifted % self.domain_width
        # print(cfg_mod)
        cfg = cfg_mod + self.domain_start
        # print(cfg_mod)
        return cfg
