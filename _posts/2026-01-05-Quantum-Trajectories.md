# Forecasting Hidden States from Noisy Data Streams

A recurring challenge across sensor analytics, financial modelling, and process control is this: 
the true state of a system cannot be observed directly. What arrives instead is a noisy, 
continuous stream of measurements that only partially reveals what is actually happening. 
The task is to estimate the hidden state in real time and update that estimate as new data arrives.

This is exactly the problem I worked on during my PhD. The system happened to be quantum 
mechanical, but the underlying mathematics is identical to Kalman filtering, particle filtering, 
and other latent-state estimation problems that appear throughout engineering and quantitative 
finance.

## [Repository](https://github.com/natwonglakhon/High-order-quantum-trajectory-simulation)

---

## The setup

The system receives one measurement per timestep. Each measurement is corrupted by *white 
noise*, so the signal-to-noise ratio at any single observation is low. The goal is to 
sequentially estimate the hidden state and propagate that estimate forward in time using 
an update map, a function that takes the current state estimate and the latest observation 
and produces a refined estimate.

Five update maps of increasing complexity were implemented and benchmarked:

| Method | Order | Description |
|---|---|---|
| Itô map | 1st | Linear update, fastest to compute |
| Rouchon-Ralph map | 2nd | Quadratic correction |
| WWC map | Higher | Wiener series expansion |
| Robinet map | 3rd | Full cubic expansion |
| Nearly exact map (Φ) | Reference | High-accuracy benchmark |

Error is measured as the average *trace distance* between each method's state estimate and 
the true hidden state across 5,000 independent realisations.

## The key insight: feature engineering from the raw signal

Standard methods summarise each observation window by its mean, discarding the temporal 
structure within the window. The $\Phi$-map uses an additional summary statistic extracted from 
the raw measurement stream: a time-weighted integral that captures how the signal evolved 
within the interval, not just its average level.

Concretely, if $y_{t_i}$ is the raw measurement record, the standard approach uses:

> $Y_t = (1/N) \sum y_{t_i}$

The $\Phi$-map additionally computes:

> $Z_{t_i} = \sum y_{t_i}(t_i - T/2) · dt$

This second statistic is cheap to compute, requires no access to higher-frequency data 
beyond what is already collected, and meaningfully reduces the estimation error across all 
realisations. It is a straightforward example of feature engineering: extracting more 
signal from the same data.

![Simulation results](https://github.com/natwonglakhon/High-order-quantum-trajectory-simulation/blob/57f9d80ac22f93108bc70391c0462bf3a75d47d0/Simulation.png?raw=true)

## Results

Across 5,000 Monte Carlo realisations over a unit time horizon, the ranking by 
time-averaged trace distance is consistent: Itô > Rouchon-Ralph ≈ WWC > Robinet > $\Phi$. 
The gap between Itô and $\Phi$ grows with timestep size, which is the expected behaviour for 
higher-order numerical methods and directly parallels the order-accuracy tradeoff in 
numerical integration.

The histogram of per-realisation errors shows that the improvement from Robinet to $\Phi$ 
is not just a shift in the mean; the $\Phi$ map also reduces the tail of large-error 
realisations, which matters in any application where worst-case accuracy is the 
binding constraint.
![Histogram results](https://github.com/natwonglakhon/High-order-quantum-trajectory-simulation/blob/57f9d80ac22f93108bc70391c0462bf3a75d47d0/Histograms.png?raw=true)

## Where this applies

The methodology generalises to any problem where:

- the true system state is latent and must be inferred from noisy observations
- measurements arrive as a continuous or high-frequency stream
- computational cost per update step is a binding constraint

Concrete applications include equipment health monitoring (inferring wear state from 
vibration signals), energy demand forecasting conditioned on real-time sensor networks, 
financial state estimation (inferring volatility regimes from tick data), and sensor 
fusion for autonomous systems where multiple noisy channels must be combined into a 
single state estimate.

---

## Research-to-industry mapping

| Research concept | Industry equivalent |
|---|---|
| Quantum trajectory | Sequential latent-state estimation |
| Continuous measurement record | Streaming sensor data |
| Hidden quantum state | Unobservable system state |
| Z statistic | Feature engineering from raw signal |
| Stochastic differential equation | Continuous-time time series model |
| Monte Carlo simulation | Ensemble probabilistic forecasting |
| Trace distance | Prediction error metric |
| Higher-order map | More accurate numerical integrator |
