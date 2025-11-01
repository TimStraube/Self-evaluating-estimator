# Self-Evaluating Estimator (SEE)

## Info

A fun little Deep Reinforcement Learning Project for experimenting with strange architectures.

## Ideas

Self-Evaluating Estimator (SEE) networks are an experimental deep reinforcement learning architecture. After each action, the previous observation from the environment is transformed into its frequency representation using the Fast Fourier Transform (FFT) algorithm. 

In addition to the classic implicit memory of a reinforcement learning agent (such as policy and value functions), SEE introduces an explicit memory module that represents a weighted superposition of past observations in the frequency domain. The weights are controlled directly by the agent through additional action heads.

Rewards are computed based on the difference between the current reward and the average memorized reward. The current reward can be thought of as the loss between the currently selected observation from memory (via parameter control) and the actual observation from the outside world. The agent's observation can be considered as the linear time-invariant system response of the self-controlled memory and the external environment observation.

![Self Evaluating Estimator Network](docs/images/SEE.png)

### Running the agent in the global environment

The global environment is the set of all local environment and transsitions inbetween.

`bash run.sh`

### Testing the logical integrity of the agent and environments

`bash test.sh`

### Generating benchmarks and visualizations of a paritcular livetime

`bash benchmark.sh`