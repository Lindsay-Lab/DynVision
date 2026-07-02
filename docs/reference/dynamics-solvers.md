# Dynamics Solvers

DynVision models the dynamics of neural activity using continuous-time differential equations. This approach aims to capture the temporal properties of biological neural networks more accurately than discrete-time recurrent models.

## The Neural Dynamics Equation

The core of DynVision's dynamics is the following differential equation:

$$
\tau \frac{dx}{dt} = -x + \Phi[f(t, r_n, r_{n-1})]
$$

Where $\tau$ is the time constant, $x$ is the neural activity, $\Phi$ is a nonlinearity, and $f$ represents the inputs to the neuron. This equation is solved numerically using one of the available solvers in DynVision.

## Available Solvers

DynVision provides two main solvers for the neural dynamics equation. When configuring a model, use the string identifier (e.g., `"euler"`, `"rk4"`), which maps to the corresponding solver class (`EulerStep`, `RungeKuttaStep`).

**Solver identifiers (config/CLI) vs class names:**

| Config String | Class Name | Method |
|---------------|------------|--------|
| `"euler"` | `EulerStep` | First-order Euler |
| `"rk4"` | `RungeKuttaStep` | 4th-order Runge-Kutta |

### 1. EulerStep

The Euler method is a first-order numerical procedure for solving ordinary differential equations (ODEs). It provides a simple approximation:

$$x(t+dt) = x(t) + \frac{dt}{\tau} \cdot [-x(t) + J(x(t))]$$

**Advantages**:

- Computational efficiency
- Memory efficiency
- Stable for small time steps

**Limitations**:

- Less accurate for rapidly changing dynamics
- Requires small time steps for stability

### 2. RungeKuttaStep

The 4th-order Runge-Kutta (RK4) method provides a more accurate approximation for the ODE:

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + \frac{dt}{2}, y_n + \frac{dt}{2}k_1)$$
$$k_3 = f(t_n + \frac{dt}{2}, y_n + \frac{dt}{2}k_2)$$
$$k_4 = f(t_n + dt, y_n + dt \cdot k_3)$$
$$y_{n+1} = y_n + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**Advantages**:

- Higher accuracy than Euler method
- Stable for larger time steps
- Better captures complex dynamics

**Limitations**:

- Computationally more expensive
- Higher memory requirements

## Parameterization of Dynamics

Both solvers use the following parameters to control the dynamics:

### 1. Time Step (dt)

The time step `dt` determines the granularity of the numerical integration. Smaller values provide more accurate solutions but require more computation.

Typical values: 1-5 ms

### 2. Time Constant (tau)

The time constant `tau` controls how quickly the neural activity responds to input. Larger values result in slower dynamics.

Typical values: 5-20 ms

### 3. Delay Parameters

DynVision implements separate, per-connection-type delays. Each delay must be an
integer multiple of `dt`. The delays are specified in milliseconds (aliases in
parentheses):

- `t_feedforward` (`tff`, $\Delta_{FF}$): feedforward delay between layers. Default `0 ms` (engineering-time unrolling).
- `t_recurrence` (`trc`, $\Delta_{RC}$): lateral recurrent delay. Default `6 ms`.
- `t_skip` (`tsk`, $\Delta_{SK}$): skip-connection delay. Default `0 ms` (engineering time).
- `t_feedback` (`tfb`, $\Delta_{FB}$): feedback-connection delay.

> **Note:** The defaults above describe *engineering-time* unrolling
> ($\Delta_{FF}=0$), where the signal propagates from input to output within a
> single timestep. In *biological-time* unrolling the feedforward delay is
> positive (e.g. `10 ms`) and the skip/feedback delays are adjusted accordingly.
> See [Engineering vs. Biological Time](../explanation/engineering-vs-biological-time.md).

## Usage in Models

The dynamics solvers are used in conjunction with recurrent connections to evolve the neural activity over time.

### Example in DyRCNNx4

In the `DyRCNNx4` model, each layer has its own dynamics solver:

```python
self.tau_V1 = torch.nn.Parameter(
    torch.tensor(self.tau, dtype=float),
    requires_grad=False,
)
self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau_V1)

# ... similar for other layers
```

During forward propagation, the dynamics solver is applied to the layer's activity:

```python
# Inside the forward method
if operation == "tstep" and hasattr(self, module_name):
    module = getattr(self, module_name)
    h = layer.get_hidden_state(-1)
    x = module(x, h)
```

## Numerical Stability

Numerical stability is a concern when using dynamics solvers, especially with nonlinear activation functions or supralinear transformations. DynVision implements stability checks that can be enabled:

```python
model = DyRCNNx4(
    stability_check=True,
    # other parameters
)
```

When enabled, the solver will check for NaN or infinite values and raise an error if detected.

## Comparison with Discrete-Time Approaches

Most RCNN models in the literature use discrete-time approaches, where recurrence is unrolled for a fixed number of steps. DynVision's continuous-time approach offers several advantages:

1. **Biological Plausibility**: Neural dynamics in the brain operate in continuous time.
2. **Temporal Resolution**: Allows for fine-grained control over temporal dynamics.
3. **Flexibility**: Different connection types can have different delays.
4. **Realistic Response Properties**: Can better capture phenomena like adaptation and subadditive temporal summation.

## Advanced Usage

### Custom Dynamics Equations

You can implement custom dynamics equations by subclassing `BaseSolver`:

```python
class CustomSolver(BaseSolver):
    def forward(self, x, h=None):
        # Implement your custom dynamics equation
        pass
```

### Heterogeneous Time Constants

Different layers can have different time constants:

```python
model = DyRCNNx4(
    tau=10.0,  # Default time constant
    # other parameters
)

# After initialization, modify specific layer time constants
model.tau_V1.data.fill_(5.0)
model.tau_V2.data.fill_(8.0)
model.tau_V4.data.fill_(12.0)
model.tau_IT.data.fill_(15.0)
```

## Performance Considerations

The dynamics solvers add computational overhead to the model, especially when using small time steps or complex solvers like RK4. Consider the following to optimize performance:

1. **Use Euler for faster computation**: The Euler method is often sufficient and much faster.
2. **Consider training with shorter sequences**: During training, shorter sequences can be used.
3. **Use mixed precision**: Enable mixed precision training for better performance.

## Biological Phenomena Captured

The dynamics solvers enable DynVision to capture several important biological phenomena:

1. **Response Latency**: Different layers show different response latencies.
2. **Adaptation**: Neural responses decrease with sustained stimulation.
3. **Subadditive Temporal Summation**: Responses to longer stimuli saturate.
4. **Contrast-Dependent Dynamics**: Responses to high-contrast stimuli are faster.
5. **Short-Term Memory**: Recurrent connections allow information persistence.


## References

For more details on the mathematical foundations of these solvers:

1. Butcher, J. C. (2016). Numerical methods for ordinary differential equations.
2. Heeger, D. J., & Mackey, W. E. (2019). Oscillatory recurrent gated neural integrator circuits (ORGaNICs), a unifying theoretical framework for neural dynamics.
3. Soo, W. W., et al. (2024). Recurrent neural network dynamical systems for biological vision.
