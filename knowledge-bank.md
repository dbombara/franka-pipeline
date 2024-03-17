# Overview of Learning-Based (RL) Control Policies

## Key Concepts
In reinforcment learning, an agent interacts with an environments $\mathcal{E}$ over a discrete number of timesteps $t \in [0,T]$. At each timestep, the agent receives a state $s_t$ and selects an action $a_t \in \mathcal{A}$ according to its policy $\pi(s_t) = a_t$. The agent then receives the next state $s_{t+1}$ and scalar reward $r_t$. 
- **Policy**: $\pi (s | a)$
- **Critic**: 
- **Actor**: The actor is the decision-maker that selects actions based on the current policy. Its responsibility is to explore the action space to maximize expected cumulative rewards. The total discounted return is:

$$
    R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

where $\gamma \in (0,1]$ is the discount factor. The goal is to maximize the expected return at each state. Define the action value as:

$$
    Q^\pi (s,a) = \mathbb{E}[R_t | s_t = s,a]
$$

to be the expected return for selecting action $a$ in state $s$ while following policy $\pi$. The optimal value function $Q^{\star}(s,a) = \max_\pi{Q^\pi (s,a)}$ give the maximum action value for state $s$ and action $a$ achievable by any policy. Similarly, the value of state $s$ under policy $\pi$ is defined as $V^\pi (s) = \mathbb{E}[R_t | s_t = s]$, and is the expected return for following policy $\pi$ from state $s$. 

## RL Algorithms

### [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- on-policy algorithm.
- action space: discrete or continuous

### [Advantage Actor Critic (A2C)](https://arxiv.org/abs/1602.01783v2)

### [Deep Q-Networks (DQN)](https://arxiv.org/pdf/1312.5602v1.pdf)

### [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290)

### [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)

## Generative Models

### [Action Diffusion Policy](https://arxiv.org/pdf/2303.04137.pdf)

Starting from $\mathbf{x}^K$ sampled from Gaussian noise, the DDPM performs $K$ iterations of denoising to produce a series of intermediate actions with decreasings levels of noise, $\mathbf{x}^k$, $\mathbf{x}^{k-1}$, ..., $\mathbf{x}^0$, until the desired noise-free output $\mathbf{x}^0$ is formed. The process follows this equation:

$$
    \mathbf{x}^{k-1} = \alpha (\mathbf{x}^k - \gamma \epsilon_\theta (\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2 I))
$$

where $\epsilon_\theta$ is the noise prediction network with parameters $\theta$ that will be optimized through learning and $\mathcal{N}(0, \sigma^2 I)$ is Gaussian noise added at each iteration.



## Online Learning Algorithms

### Key Concepts

TODO: Add definition of regret bound.

### Follow the Regularized Leader (FTRL)


### Online Gradient Descent (OGD)

### Online Mirror Descent (OMD)

### Mechanic (Learning Rate Tuner)