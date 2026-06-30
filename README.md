# Modeling and Physics-Enhanced Fault Detection in Wastewater Pump Stations

This repository contains the code accompanying the paper
"Modeling and Physics-Enhanced Fault Detection in Wastewater Pump Stations".

The code implements a parameterized wastewater pump station simulator and diagnostic methods for fault detection and fault-origin isolation. The simulator models transient sump hydraulics, pump and system curves, VFD/soft-start operation, sensor noise, and representative pump-side and system-side faults such as internal blockage and pipe clogging.

## Repository structure

- `Blockage/`: pump internal blockage simulations
- `Clogging/`: pipe clogging/system-side fault simulations
- `Diagnostic methods/`: fault detection and isolation algorithms
- `ECDF/`: inflow/statistical modeling scripts
- `Sensitivity Analysis/`: parameter sensitivity experiments
- `Simulation_verification/`: simulator validation scripts
- `Simulation_verification(current-power)/`: power-related validation
- `Unseen Scenarios/`: additional test scenarios
