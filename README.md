# Coordination Dynamics in a Networked Stag-Hunt Model

This repository contains the code, data, and report for a project on coordination dynamics in a networked Stag-Hunt model, completed for the **Model-Based Decision Making** course.

## Overview

The project studies how coordination outcomes depend on:
- Initial adoption and payoff incentives
- Network topology (grid, small-world, Erdős–Rényi, Barabási–Albert)
- Policy interventions and their timing

An agent-based evolutionary model with infrastructure feedback is used to study tipping behavior and policy leverage.

## Model and Policies

Agents choose between **adopt** and **not adopt**, interacting with neighbors on a network. Payoffs depend on local coordination and a global infrastructure variable. Strategies update via a logit best-response rule, producing stochastic tipping near critical thresholds.

Two policy types are evaluated:
- **Temporary subsidies:** time-limited payoff increases  
- **Targeted seeding:** forcing a fraction of agents to adopt

Policies are assessed using coordination probability, final adoption, and cost-efficiency.

## Repository Structure

- `phase_sweep_code/` – code used for phase sweep and sensitivity
- `policy_code` - code used for policy and network analysis 
- `**/plots/` – figures used in the report and slides  
- `report/` – PDF of the report
- `presentation/` – PDF of slides  

## Results (Summary)

- Coordination is nonlinear and path-dependent
- Network structure shifts tipping thresholds and reliability
- Subsidies can be effective but are timing-sensitive and costly
- Targeted seeding is consistently more cost-efficient

## Author

Joeri Fresen  
University of Amsterdam
