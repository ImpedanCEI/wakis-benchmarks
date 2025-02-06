# Wakis Benchmarking Repository

## Overview

This repository is dedicated to benchmarking Wakis against commercial and open-source electromagnetic simulation tools such as CST Wakefield Studio and WarpX. The goal is to validate and compare wakefield simulations performed using Wakis, ensuring accuracy, efficiency, and reliability in different scenarios.

## About Wakis

`Wakis` is a 3D Time-domain Electromagnetic solver that solves the Integral form of Maxwell's equations using the Finite Integration Technique (FIT) numerical method. It computes the longitudinal and transverse wake potential and beam-coupling impedance from the simulated electric and magnetic fields. It is also a multi-purpose solver, capable of simulating planewaves interaction with nano-structures, optical diffraction, and much more!

* Check the source repository: https://github.com/ImpedanCEI/wakis
* Check the notebook examples: https://github.com/ImpedanCEI/wakis/tree/main/notebooks
* Check the documentation: https://wakis.readthedocs.io/en/latest/index.html 

## Benchmarking Goals

* Accuracy Validation: Compare simulation results from Wakis against CST Wakefield Studio and WarpX.

* Performance Analysis: Evaluate computational efficiency, runtime, and resource usage.

* Parameter Studies: Test various configurations (e.g., different boundary conditions, mesh resolutions, and excitation signals).

* Code Reproducibility: Ensure that Wakis provides consistent results under different conditions.