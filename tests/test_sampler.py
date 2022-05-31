import numpy as np
import pytest
from interpmcmc.sampler import PeriodicityEnforcer


class TestPeriodicityEnforcer():
    @pytest.mark.parametrize("x", (-.9, .1, 1.1))
    @pytest.mark.parametrize("y", (-.9, .1, 1.1))
    def test_0_1(self, x, y):
        cfg = np.array([x, y])

        domain = np.array([[0, 1], [0, 1]])
        PBC = PeriodicityEnforcer(domain)

        transformed_cfg = PBC(cfg)
        assert np.allclose(transformed_cfg, np.array([.1, .1]))

    @pytest.mark.parametrize("x", (-2.9, -.9, 1.1))
    @pytest.mark.parametrize("y", (-2.9, -.9, 1.1))
    def test_minus_1_1(self, x, y):
        cfg = np.array([x, y])
        domain = np.array([[-1, 1], [-1, 1]])
        PBC = PeriodicityEnforcer(domain)

        transformed_cfg = PBC(cfg)
        assert np.allclose(transformed_cfg, np.array([-.9, -.9]))
