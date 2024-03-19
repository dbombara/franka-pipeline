# Overview of Learning-Based (RL) Control Policies

## Key Concepts
In reinforcment learning, an agent interacts with an environment $\mathcal{E}$ over a discrete number of timesteps $t \in [0,T]$. At each timestep, the agent receives a state $s_t$ and selects an action $a_t \in \mathcal{A}$ according to its **policy** $\pi(s, a)$. The agent then receives the next state $s_{t+1}$ and scalar reward $r_t$. The **actor** is the decision-maker that selects actions based on the current policy. Its responsibility is to explore the action space to maximize expected cumulative rewards. The total discounted return is:

$$
    R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

where $\gamma \in (0,1]$ is the discount factor. The goal is to maximize the expected return at each state. Define the action value as:

$$
    Q^\pi (s,a) = \mathbb{E}[R_t | s_t = s,a]
$$

to be the expected return for selecting action $a$ in state $s$ while following policy $\pi$. The optimal value function $Q^{\star}(s,a) = \max_\pi{Q^\pi (s,a)}$ give the maximum action value for state $s$ and action $a$ achievable by any policy. Similarly, the value of state $s$ under policy $\pi$ is defined as $V^\pi (s) = \mathbb{E}[R_t | s_t = s]$, and is the expected return for following policy $\pi$ from state $s$. 

## RL Algorithms

### [Vanilla Policy Gradient (VPG)](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

- On-policy algorithm.
- Action space: discrete or continuous.

Let $\pi_\theta$ be a policy with parameters $\theta$. Let $J(\theta_\pi)$ denote the expected finite-horizon undiscounted return of the policy. The gradient of $J(\theta_\pi)$ is given by:

$$
∇_{\theta_\pi} J(\theta_\pi) = \mathbb{E}_{\tau \sim \pi_{\theta_\pi}} \left[ \sum_{t=0}^T \nabla_{\theta_\pi} \log \pi_{\theta_\pi} (a_t | s_t) A^{\pi_\theta}(s_t, a_t) \right]
$$

where $\tau$ is a trajectory and $A^{\pi_\theta}(s_t, a_t)$ is the advantage function. The advantage function is the difference between the action value and the value function:

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

### [Online Convex Optimization (OCO)](https://zhiyuzz.github.io/Dissertation_Zhiyu.pdf)

Online convex optimization (OCO) is a repeated game between a learning agent and an adversarial environment $\mathcal{E}$. In each round $t$:
1. The agent selects a decision $x_t \in \mathcal{X}$ based on its observations from previous rounds. $\mathcal{X}$ is a nonempty, closed, and convex subset of $\mathbb{R}^d$.
2. The environment selects a convex cost function $l_t: \mathcal{X} \rightarrow \mathbb{R}$, which deterministically depends on the prediction history of the agent, $x_1 \dotsc, x_t$.
3. The agent incurs a loss $l_t(x_t)$ and observes it.

The game ends after $T$ rounds, at which point the agents' cumulative loss is compared to an alternative prediction sequence $u_1, \dotsc, u_t \in \mathcal{X}$, which is referred to as the **comparator sequence**.

The goal of the agent is to minimize the *regret*, defined as the difference between the cumulative loss of the agent's predictions and the cumulative loss of the comparator:

$$
    \texttt{Regret}_T (\mathcal{E}, u_{1:T}) := \sum_{t=1}^T l_t(x_t) - \sum_{t=1}^T l_t(u_t).
$$

However, neither the environment $\mathcal{E}$ nor the the comparator sequence $u_1, \dotsc, u_T$ is not known to the agent.

### [Follow the Regularized Leader (FTRL)](https://arxiv.org/pdf/1912.13213.pdf)

### [Online Gradient Descent (OGD)](https://arxiv.org/pdf/1912.13213.pdf)

### [Online Mirror Descent (OMD)](https://arxiv.org/pdf/1912.13213.pdf)

### [Mechanic (Learning Rate Tuner)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/955499a8e2860ed746717c1374224c43-Abstract-Conference.html)

## References