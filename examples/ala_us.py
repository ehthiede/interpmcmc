import numpy as np
from interpmcmc import InterpForceField, HarmonicPotential, Compose
from interpmcmc.sampler import simulate_walker, PeriodicityEnforcer
from interpmcmc.umbrella_sampling import construct_centers_on_grid
from interpmcmc.umbrella_sampling import fk_from_stdev
import argparse


def _parse_CLAs():
    parser = argparse.ArgumentParser(description="Run umbrella sampling \
            calculation on a saved alanine dipeptide pmf")
    parser.add_argument('pmf', type=str, help='Numpy binary of the \
            two-dimensional PMF to load from file')
    parser.add_argument('window_idx', type=int, help='Which window on a \
            20 x 20 grid to sample.')
    parser.add_argument('--output', type=str, default=None, help='Where to \
            save the output of the simulation')
    parser.add_argument('--nsteps', type=int, defaulte=1e5, help='How many \
            steps to run the calculation for')
    parser.add_argument('--burnin', type=int, defaulte=1e3, help='Number of \
            steps to burn in the trajectory for.')
    parser.add_argument('--dt', type=float, defaulte=1e-3, help='Overdamped \
            Langevin Timestep')
    return parser


def build_window_forcefield(pmf, domain, center, fk):
    unbiased_ff = InterpForceField.from_npy_pmf(pmf, domain)
    bias_fxn = HarmonicPotential(fk, center, domain=domain)
    FF = Compose(unbiased_ff, bias_fxn)
    return FF


def define_windows(domain):
    """
    Defines the grid of umbrella sampling windows
    """
    L = 20
    hw = (domain[0][1] - domain[0][0])/L
    centersx = np.linspace(domain[0][0]+hw/2, domain[0][1]-hw/2, L)
    centersy = np.linspace(domain[1][0]+hw/2, domain[1][1]-hw/2, L)
    centers = construct_centers_on_grid(centersx, centersy)
    fk = fk_from_stdev(hw / 2.)
    return centers, fk


def main():
    # Parse basic
    parser = _parse_CLAs()
    args = parser.parse_args()
    output_str = args.output
    if output_str is None:
        output_str = "window_%d" % args.window_idx

    # Load raw PMF and define metadata
    ala_pmf = np.load(args.pmf)
    domain = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
    centers, fk = define_windows(domain)
    center_i = centers[args.window_idx]

    FF = build_window_forcefield(ala_pmf, domain, center_i, fk)
    PBC = PeriodicityEnforcer(domain)

    samples = simulate_walker(center_i, FF, PBC, nsteps=args.nsteps,
                              dt=args.dt, burnin=args.burnin)
    np.save(output_str, samples)


if __name__ == "__main__"():
    main()
