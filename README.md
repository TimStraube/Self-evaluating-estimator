# Self evaluating estimator (SEE)

## Info

A fun little Deep Reinforcement Learning Project for experiementing with strange architectures.

## Ideas

Self evaluating estimator networks are a experimental deep reinforcement learning architecture. Post action the previous observation from the environment is being transformed into frequency domain using the fast fourier transform algorithm.

![Self Evaluating Estimator Network](docs/images/SEE.png)

## Getting started

### Installing poetry

### Running the agent in the global environment

The global environment is the set of all local environment and transsitions inbetween.

`bash run.sh`

### Testing the logical integrity of the agent and environments

`bash test.sh`

### Generating benchmarks and visualizations of a paritcular livetime

`bash benchmark.sh`