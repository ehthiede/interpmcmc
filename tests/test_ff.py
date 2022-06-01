import pytest
import numpy as np
from interpmcmc.ff import Compose, HarmonicPotential


@pytest.mark.parametrize("x", (0.0, 1.0, -.5))
def test_compose_on_harmonics(x):
    H1 = HarmonicPotential(1., 0.0)
    H2 = HarmonicPotential(2., 0.0)

    H_sum = Compose([H1, H2])

    FF_x = H_sum(x)  # Should be -3 x
    assert(np.abs(FF_x + 3 * x) < 1e-6)


class TestHarmonicPotential():
    @pytest.mark.parametrize("ndim", range(1, 4))
    @pytest.mark.parametrize("center", [0.0, 1.0])
    def test_harmonic_force(self, ndim, center):
        x = np.random.randn(ndim)
        H = HarmonicPotential(1., center)

        force = H(x)
        assert(np.allclose(force, -1 * (x - center)))
    
    @pytest.mark.parametrize("x_shift", [-2, 0, 2])
    @pytest.mark.parametrize("y_shift", [-2, 0, 2])
    def test_w_minimage(self, x_shift, y_shift):
        domain = np.array([[-1., 1.], [-1., 1.]])
        config = np.array([0.1, - 0.1])
        center = np.array([0.1, 0.1])

        config_shift = np.copy(config)
        config_shift[0] += x_shift
        config_shift[1] += y_shift
        H = HarmonicPotential(1., center, domain=domain)

        force = H(config_shift)
        assert(np.allclose(force, -1 * (config - center)))
