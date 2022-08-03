import numpy as np
from interpmcmc import InterpForceField, HarmonicPotential, Compose
from interpmcmc.sampler import metropolis, PeriodicityEnforcer
from interpmcmc.umbrella_sampling import construct_centers_on_grid
from interpmcmc.umbrella_sampling import fk_from_stdev
from tqdm import tqdm
import argparse


def _parse_CLAs():
    parser = argparse.ArgumentParser(description="Run umbrella sampling \
            calculation on a saved alanine dipeptide pmf")
    parser.add_argument('pmf', type=str, help='Numpy binary of the \
            two-dimensional PMF to load from file')
    parser.add_argument('--replicate', type=int, default=0, help='index of \
            statistical replicate.')
    parser.add_argument('--temp', type=float, default=None, help='Temperature\
            if none, chosen so kT=1..')
    parser.add_argument('--num-windows', type=int, default=20, help='number of\
            windows to use')
    parser.add_argument('--force-constant', type=float, default=None, help='force\
            constant to use')
    parser.add_argument('--output', type=str, default=None, help='Where to \
            save the output of the simulation')
    parser.add_argument('--nsteps', type=int, default=1e4, help='How many \
            steps to run the calculation for')
    parser.add_argument('--burnin', type=int, default=1e3, help='Number of \
            steps to burn in the trajectory for.')
    parser.add_argument('--dx', type=float, default=0.1, help='Metropolis \
            Step Size')
    return parser


def build_window_forcefield(pmf, domain, center, fk):
    unbiased_ff = InterpForceField.from_npy_pmf(pmf, domain)
    bias_fxn = HarmonicPotential(fk, center, domain=domain)
    FF = Compose([unbiased_ff, bias_fxn])
    return FF


def define_windows(domain, L, fk=None, shift_centers_by_hw=False):
    """
    Defines the grid of umbrella sampling windows
    """
    hw = (domain[0][1] - domain[0][0])/L
    if shift_centers_by_hw:
        centersx = np.linspace(domain[0][0]+hw/2, domain[0][1]-hw/2, L)
        centersy = np.linspace(domain[1][0]+hw/2, domain[1][1]-hw/2, L)
    else:
        centersx = np.linspace(domain[0][0], domain[0][1], L,endpoint=False)
        centersy = np.linspace(domain[1][0], domain[1][1], L,endpoint=False)
    centers = construct_centers_on_grid(centersx, centersy)
    if fk is None:
        fk = fk_from_stdev(hw / 2.)
    return centers, fk


def main():
    # Parse basic
    parser = _parse_CLAs()
    args = parser.parse_args()

    # Load raw PMF and define metadata
    T = args.temp
    if T is None:
        kT = 1.
    else:
        k_B =  8.314462618E-3
        kT = k_B * T
    ala_pmf = np.load(args.pmf)
    domain = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
    print(args.force_constant, 'args.fc')
    centers, fk = define_windows(domain, args.num_windows, args.force_constant)
    print(centers)
    print(fk, 'out fk')
        
    output_str = args.output
    if output_str is None:
        output_str = "traj_data/rep_%d.npy" % args.replicate
    
    all_trajs = []
    for i in tqdm(range(len(centers))):
        center_i = centers[i]
        FF = build_window_forcefield(ala_pmf, domain, center_i, fk)
        PBC = PeriodicityEnforcer(domain)

        samples = metropolis(center_i, FF, PBC, nsteps=args.nsteps,
                             dx=args.dx, burnin=args.burnin, kT=kT)
        all_trajs.append(samples)

    np.save(output_str, all_trajs)


if __name__ == "__main__":
    main()
