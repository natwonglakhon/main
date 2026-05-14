2026-01-05-Quantum-Trajectories.md
# High-Order Quantum Trajectory Simulation

A Python codebase for simulating and benchmarking quantum trajectory methods using higher-order stochastic integration schemes. The project reproduces numerical results from research on high-order quantum trajectory reconstruction, comparing five distinct simulation maps via trace-distance error analysis.

---

## Overview

Quantum trajectory simulation involves reconstructing a quantum state conditioned on a continuous measurement record. This project implements and compares five maps of varying order:

| Method | Label | Description |
|---|---|---|
| Itô map | `I` | First-order stochastic integration |
| Rouchon–Ralph map | `RR` | Second-order correction map |
| WWC map | `W` | Higher-order Wiener expansion |
| Robinet map | `RB` | Third-order stochastic map |
| Nearly exact map | `Φ` | High-accuracy reference map |

Error is quantified by the **trace distance** between a simulated conditioned state and the true quantum trajectory.

![Quantum trajectory simulations](https://github.com/natwonglakhon/main/blob/master/images/Simulation_prev/Qtraj_prev.png?raw=true)
Overall flights.

---

## [Go to Repository](https://github.com/natwonglakhon/High-order-quantum-trajectory-simulation)

