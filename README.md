This repository contains the sound change
models for "Modeling the interface between phonetic biases, synchronic category learning, and language change".
It uses infinite Mixture of Gaussians models, implemented
by SciKit's Bayesian Mixture of Gaussians.
The focus is on modeling the development of nasal vowel allophony
from nasal vowel coarticulation.

Relevant files:

simulations/learner_definitions.py: Defines the wrapper classes for SciKit's model to make running agent-based simulations cleaner. Includes a Category, Language, and Learner classes.

simulations/dirichlet_hyperparam_search.py: Runs instances of Learner agents with different input languages and hyperparameter settings, and outputs evaluation metrics to csv files in the simulations/parameter_search_outputs directory.

simulations/mnmc_sim.py: (in progress) - Simulates learning across generations with a bias to make low vowel exemplars more nasal in general. Plots are output to simulations/mnmc_plots

simulations/input_overlap.py: Demonstration of how the number of learned categories depends on the amount of overlap in the generating/parent distributions. Outputs in simulations/overlap_simulation_outputs

simulations/simulated_input_distributions: Plots of the starting distributions used in simulations.

simulations/test_learner_definitions.py: (in progress) Unit tests for the Category, Language, and Learner components


