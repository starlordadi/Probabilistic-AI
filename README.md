# Probabilistic-AI
Implementation of Bayesian Learning and Reinforcement Learning algorithms for the course Probabilistic Artificial Intelligence at ETH Zuerich. The algorithms encompass different areas of Bayesian Machine Learning like,
  1. Gaussian Processes
  2. Bayesian Neural Networks
  3. Bayesian Optimisation
  4. Reinforcement Learning

## Gaussian Processes
Implemented Gaussian Process Regression for modelling and inferring fine particle concentration at different places to model air pollution. Used ``` scikit-learn ``` and ``` gpytorch ``` libraries to implement the Gaussian Processes (GP) and validated on a sample dataset provided. Tested different GP kernels and their combinations ideal for the data and implemented a modified inference method to handle asymmetric costs.

## Bayesian Neural Networks
Implemented the following algorithms for multi-class classification task on the MNIST fashion dataset,
  1. Monte Carlo Dropout
  2. Ensemble Learning
  3. Stochastic Gradient Langevin Dynamics (SGLD)
  
## Bayesian Optimisation
Implemented a Bayesian Optimisation algorithm with a constrained Acquisition Function to handle inequality constraints on the domain. Tested the algorithm on a toy problem in 2D.

## Reinforcement Learning
Implemented a model free actor-critic algorithm based on Generalised Advantage Estimation for the Lunar Lander task. Trained and validated the policy in simulation. 

![lunar_lander](https://user-images.githubusercontent.com/36773602/231979794-384cd721-a212-4053-bcb8-2a3941e5a0c5.gif)
