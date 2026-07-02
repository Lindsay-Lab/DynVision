"""
Modules to stepwise solve the dynamical systems differential equation:
$$$
\tau \frac{dx}{dt} = -x + \frac{dt}{tau} * [-x + W(x)]
$$$
to evolve the activity tensor from x(t) to x(t+dt) with a given time step dt and time constant tau.
"""

from typing import Optional, Union, Tuple
import logging

import torch
from pytorch_lightning import LightningModule


logger = logging.getLogger(__name__)

__all__ = ["EulerStep", "RungeKuttaStep"]


class BaseSolver(LightningModule):
    """Base class for ODE solvers with common utilities."""

    def __init__(
        self,
        dt: float = 2.0,
        tau: Union[float, torch.nn.Parameter] = 5.0,
    ) -> None:
        super().__init__()

        if isinstance(tau, (int, float)):
            self.tau = torch.nn.Parameter(torch.tensor(tau), requires_grad=False)
        else:
            self.tau = tau

        self.dt = dt

        self.reset()

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Reset hidden state."""
        self.hidden_state = None


class EulerStep(BaseSolver):
    """
    Euler integration step for neural dynamics:
    $$$
    x(t+dt) = x(t) + dt/tau * [-x(t) + W(x(t))]
    $$$

    Args:
        dt: Integration time step
        tau: Time constant (can be trainable parameter)
    """

    def __init__(
        self,
        dt: float = 2.0,
        tau: Union[float, torch.nn.Parameter] = 5.0,
    ) -> None:
        super().__init__(dt=dt, tau=tau)

    def forward(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform one Euler integration step.

        Args:
            x: Input tensor W(x(t))
            h: Hidden state x(t) (optional)

        Returns:
            Updated state x(t+dt)
        """
        if x is None:
            return None

        use_own_hidden_state = h is None
        if use_own_hidden_state and self.hidden_state is not None:
            h = self.hidden_state

        # Compute update
        if h is None:
            y = self.dt / self.tau * x
        else:
            y = h + self.dt / self.tau * (x - h)

        if use_own_hidden_state:
            self.hidden_state = y

        return y


class RungeKuttaStep(BaseSolver):
    """
    4th order Runge-Kutta integration step for improved accuracy:

    Features same optimizations as EulerStep with higher numerical precision.
    """

    def forward(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Perform one Runge-Kutta integration step."""
        if x is None:
            return None

        use_own_hidden_state = h is None
        if use_own_hidden_state and self.hidden_state is not None:
            h = self.hidden_state

        if h is None:
            y = self.dt / self.tau * x
        else:
            # RK4 integration
            k1 = (x - h) / self.tau
            k2 = (x - (h + self.dt * k1 / 2)) / self.tau
            k3 = (x - (h + self.dt * k2 / 2)) / self.tau
            k4 = (x - (h + self.dt * k3)) / self.tau
            y = h + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        if use_own_hidden_state:
            self.hidden_state = y

        return y


def test_solver(
    solver_class: LightningModule,
    dt: float = 0.1,
    tau: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """
    Test solver implementation with various input conditions.

    Args:
        solver_class: Solver class to test
        dt: Integration time step
        tau: Time constant
        device: Computation device
    """
    inf = 1000
    rtol = 1e-03
    atol = 1e-04

    solver_step = solver_class(dt, tau)
    logger.info(f"Testing {solver_class.__name__} with dt={dt}, tau={tau}")

    # Test constant input
    c = torch.tensor([1.0], device=device)
    logger.info(f"Testing constant input c={c.item()}")
    for _ in range(inf):
        y = solver_step(c)
    assert torch.allclose(y, c, rtol=rtol, atol=atol)

    solver_step.reset()

    # Test exponential decay
    k = 0.7
    logger.info(f"Testing exponential decay k={k}")
    x_0 = torch.tensor([1.0], device=device)
    x_1 = torch.tensor([0.8], device=device)
    y = torch.tensor([0.0], device=device)
    x = solver_step(x_1, x_0)
    for _ in range(inf):
        x = solver_step(k * x)
    assert torch.allclose(x, y, rtol=rtol, atol=atol)

    solver_step.reset()

    # Test None input
    assert solver_step(None) is None

    # Test numerical stability
    try:
        solver_step(torch.tensor([float("inf")], device=device))
        assert False, "Should raise stability error"
    except ValueError:
        pass

    logger.info("All tests passed!")


if __name__ == "__main__":
    # Test both solvers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for solver_class in [EulerStep, RungeKuttaStep]:
        test_solver(solver_class, device=device)
