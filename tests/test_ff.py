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
