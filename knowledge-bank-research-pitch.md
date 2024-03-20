# Research Pitch: Parameter-Free Online Finetuning of Deep Visuomotor Policies

TLDR: Use the Mechanic algorithm to select learning rates during online reinforcement learning (RL) of visuomotor policies on the Franka Research 3.

The [Mechanic learning rate tuner](https://proceedings.neurips.cc/paper_files/paper/2023/file/955499a8e2860ed746717c1374224c43-Paper-Conference.pdf) is described below:

# Knowledge Bank

## Reinforcement Learning Concepts

A state $s$ is a complete description of the system, whereas an observation $o$ is a partial description of the system. The action space is the set of valid actions, and may be continuous or discrete.

A **policy** is a mapping from state to action. It may be determinstic:

$$
a_t = \mu (s_t)
$$

in which case the policy is a function $\mu: \mathcal{S} \rightarrow \mathcal{A}$, or stochastic:

$$
a_t \sim \pi(\cdot | s_t)
$$

in which case the policy is a conditional probability distribution $\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$. The policy may be parameterized by a set of parameters $\theta$.

$$
\begin{align*}
    a_t = \mu_\theta (s_t) \\
    a_t \sim \pi_\theta (\cdot | s_t)
\end{align*}
$$

An example of a deterministic policy is a multi-layer perceptron (MLP). Stochatic policies are commonly *categorial policies* or *diagonal Gaussian policies*. Categorial policies may be used in discrete action spaces, but diagonal Gaussian policies must be used in continuous action spaces.

### Trajectories and State Transitions
A trajectory $\tau$ is a sequence of states and actions in the system:

$$
\tau = (s_0, a_0, s_1, a_1, \dotsc, s_T, a_T)
$$

where $T$ is the final time. The initial state $s_0$ is sampled from the state-state distribution:

$$
s_0 \sim \rho_0(\cdot)
$$

State transitions may be deterministic or stochastic. In the deterministic case, the next state is a function of the current state and action:

$$
s_{t+1} = f(s_t, a_t)
$$

For the stochastic case, the next state is a random variable:

$$
    s_{t+1} \sim P(\cdot | s_t, a_t)
$$

where $P$ is the state transition probability distribution.

### Return and Reward

The reward function $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ maps states and actions to scalar rewards:

$$
r_t = R(s_t, a_t, s_{t+1})
$$

but may be simplified to $r_t = R(s_t, a_t)$ or $r_t = R(s_t)$. The goal of the agent is to maximize the cumulative reward (the return), which may be defined in different ways. The *finite-horizon undiscounted return* is:

$$
R(\tau) = \sum_{t=0}^T r_t
$$

Another type of return is the *infinite-horizon discounted return*:

$$
R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
$$

wgere $\gamma \in (0,1]$ is the discount factor. The discount factor is used to ensure that the return is finite in the infinite-horizon case. The discount factor also allows the agent to weigh immediate rewards more heavily than future rewards.

### RL Problem Definition

Suppose that both the environment transitions and policy are stochastic. The probability of a trajectory $\tau$ over $T$ steps is:

$$
    P(\tau | \pi) = \rho_0 (s_0) ∏_{t=0}^{T-1} \pi(a_t | s_t) P(s_{t+1} | s_t, a_t)
$$

The expected return for any reward function $R$ is:

$$
\begin{align*}
    J(\pi) &= \int_\tau P(\tau | \pi) R(\tau) \\
    &=\mathbb{E}_{\tau \sim P(\tau | \pi)} [R(\tau)].
\end{align*}
$$

The main optimization problem is RL is to find:

$$
    \pi^{\star} = \arg \max_{\pi} J(\pi)
$$

where $\pi^{\star}$ is the optimal policy.

### Value Functions

The **on-policy value function**, $V^\pi (s)$,  gives the expected return starting from state $s$ and following policy $\pi$:

$$
    V^\pi (s) = \mathbb{E}_{\tau \sim \pi} [R (\tau) | s_0 = s]
$$

The **on-policy action-value function**, $Q^\pi (s,a)$, gives the expected return starting from state $s$, taking arbitrary action $a$, and then following policy $\pi$:

$$
    Q^\pi (s,a) = \mathbb{E}_{\tau \sim \pi} [R (\tau) | s_0 = s, a_0 = a]
$$

The **optimal value function**, $V^\star (s)$, gives the expected return starting from state $s$ and following the optimal policy $\pi^\star$:

$$
\begin{align*}
    V^\star (s) &= \max_\pi V^\pi (s) \\
    &= \max_\pi \mathbb{E}_{\tau \sim \pi} [R (\tau) | s_0 = s]
\end{align*}
$$

The **optimal action-value function**, $Q^\star (s,a)$, gives the expected return starting from state $s$, taking arbitrary action $a$, and then following the optimal policy $\pi^\star$:

$$
\begin{align*}
    Q^\star (s,a) &= \max_\pi Q^\pi (s,a) \\
    &= \max_\pi \mathbb{E}_{\tau \sim \pi} [R (\tau) | s_0 = s, a_0 = a]
\end{align*}
$$

If $Q^\star (s,a)$ is known, then the optimal action is:

$$
a^\star = \arg \max_a Q^\star (s,a)
$$

### Bellman Equations

The basic idea of the Bellman Equations is "The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next". The Bellman equations for on-policy value  functions are:

$$
\begin{align*}
    V^\pi (s) &= \mathop{\mathbb{E}}_{a \sim \pi, s' \sim P} \left[r(s,a) + \gamma V^\pi (s') \right] \\
    Q^\pi (s,a) &= \mathop{\mathbb{E}}_{s' \sim P} \left[r(s,a) + \gamma \mathop{\mathbb{E}}_{a' \sim \pi} [Q^\pi (s', a')] \right],
\end{align*}
$$

where $s' \sim P := s' \sim P(\cdot | s,a)$. Similarly, $a \sim \pi$ is shorthand for s$a \sim \pi(\cdot | s)$. The Bellman equations for the optimal value functions are:

$$
\begin{align*}
    V^\star (s) &= \max_a \mathop{\mathbb{E}}_{s' \sim P} \left[r(s,a) + \gamma V^\star (s') \right], \\
    Q^\star (s,a) &= \mathop{\mathbb{E}}_{s' \sim P} \left[r(s,a) + \gamma \max_{a'} Q^\star (s', a') \right].
\end{align*}
$$

### Advantage Functions

The advantage function $A^\pi (s,a)$ for policy $\pi$ describes the relative advantage of taking action $a$ in state $s$, over randomly selecting an action according to the policy $\pi(\cdot | s)$. The advantage function is defined as:

$$
    A^\pi (s,a) = Q^\pi (s,a) - V^\pi (s)
$$

### RL Algorithms

RL algorithms can be model-based or model-free. Model-free algorithms can be further divided into policy optimization and Q-learning.

#### Policy Optimization

Methods in this catego represent a policy explicity as $\pi_\theta (a | s)$. The parameters $\theta$ are optimized directly via gradient ascent on the objective $J(\pi_\theta)$, or indirectly via local approximations of $J(\pi_\theta)$. The optimization is typically performed *on policy*, meaning each update only uses data collected while acting according to the most recent version of the policy.  Policy optimization typically involves learning an approximator $V_\phi (s)$ for the value function $V^\pi (s)$. Some examples of policy optimization algorithms are A2C/A3C and PPO.

#### Q-Learning

Methods in this family learn an approximator $Q_\theta (s,a)$ for the optimal action-value function $Q^\star (s,a)$. Typically an objective function based on the Bellman equation is used. The optimization is typically performed *off policy*, meaning that the data used to update the approximator can be collected by any policy. Some examples of Q-learning algorithms are DQN and C51.

#### Interpolating between Policy Optimization and Q-Learning

Some examples are DDPG and SAC.

### Model-Based RL

#### Background: Pure Planning

#### Expert Iteration

#### Data-Augmentation for Model-Free Methods

#### Embedding Planning Loops into Policies

## [Action Diffusion Policy](https://arxiv.org/pdf/2303.04137.pdf)

### DDPM

Starting from $\mathbf{x}^K$ sampled from Gaussian noise, the DDPM performs $K$ iterations of denoising to produce a series of intermediate actions with decreasings levels of noise, $\mathbf{x}^k$, $\mathbf{x}^{k-1}$, ..., $\mathbf{x}^0$, until the desired noise-free output $\mathbf{x}^0$ is formed. The process follows this equation:

$$
    \mathbf{x}^{k-1} = \alpha (\mathbf{x}^k - \gamma \epsilon_\theta (\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2 I))
$$

where $\epsilon_\theta$ is the noise prediction network with parameters $\theta$ that will be optimized through learning and $\mathcal{N}(0, \sigma^2 I)$ is Gaussian noise added at each iteration. The above Eq. (1) may be interpreted as a single noisy gradient descent step:

$$
    \mathbf{x}^{\prime} = \mathbf{x} - \gamma \nabla E(\mathbf{x})
$$

where $\gamma$ is the learning rate. The nosie prediction network $\epsilon_\theta$ is used to estimate the gradient $\nabla E(\mathbf{x})$. The choices of $\alpha$, $\gamma$, and $\sigma$ as functions of $k$ are analogous to the learning rate schedule in gradient descent. Previous work showed that $\alpha$ slighty less than 1 is ideal.

### DDPM Training

The training process starts by randomly drawings umodified examples $\mathbf{x}^0$ from the dataset. For each sample, a denoising iteration $k$ is randomly selected. Then, a random noise $ɛ^k$ with appropriate variance for iteration $k$ is slected. The noise prediction network $\epsilon_\theta$ is trained to minimize the difference between the predicted noise and the actual noise in terms of the mean-squared error. The loss function $\mathcal{L}$ is then:

$$
    \mathcal{L} = MSE (\epsilon^k, \epsilon_\theta (\mathbf{x}^k, k))
$$

Previously work showed that minimizing the loss function above also minimizes the variational lower bound of the KL-divergence between the data distribution $p(\mathbf{x})$ and the distribution of samples drawn from the DDPM $q(\mathbf{x^0})$.

### DDPM for Learning Visuomotor Policies

In the diffusion model paper, the standard DDPM was modified such that:

1. The output $\mathbf{x}$ represents robot actions $\mathbf{A}$.
2. The denoising process is conditions on input observation $\mathbf{O}_t$.

#### Closed-Loop Action-Sequence Prediction

At timestep $t$, the policy takes the latest $T_O$ steps of observation data $\mathbf{O}_t$ as input and predicts $T_p$ steps of actions. However, only $T_a < T_p$ steps of actions are executed on the robot without replanning. $T_O$ is defined as the *observation horizon*, $T_p$ is the *action prediction horizon*, and $T_a$ is the *action execution horizon*.

#### Visual Observation Conditioning

A DDPM is used to approximate the conditional distribution $p(\mathbf{A}_t | \mathbf{O}_t)$ (instead of the joint distribution $p(\mathbf{A}_t,\mathbf{O}_t)$ from previous work). This "allows the model to predict actions conditioned on observations without the cost of inferring future states". The standard DDPM is then modified such that:

$$
    \mathbf{A}_{t}^{k-1} = \alpha (\mathbf{A}_{t}^{k} - \gamma \epsilon_\theta (\mathbf{A}_{t}^{k}, \mathbf{O}_{t}^{k}, k) + \mathcal{N}(0, \sigma^2 I))
$$

The training loss is then modified such that:

$$
    \mathcal{L} = MSE (\epsilon^k, \epsilon_\theta (\mathbf{A}_t^0, \mathbf{O}_t, k))
$$

#### Network Architecture

##### CNN-Based Diffusion Policy

##### Time-Series Diffusion Transformer

#### Training Stability

An implicit policy represents the action distribution using an Energy-Based Model (EBM):

$$
    p_\theta (\mathbf{A}_t | \mathbf{O}_t) = \frac{1}{Z(\mathbf{O},\theta)} \exp (-E_\theta (\mathbf{A}, \mathbf{O}))
$$

where $Z(\mathbf{O}, \theta)$ is an intractical normalization constant (with respect to $\mathbf{A}$).

To train the EBM for implicit policy, the following loss function is used:

$$
    \mathcal{L} = -\mathop{\mathrm{log}} 
    \left(\frac
            {e^{-E_\theta}(\mathbf{o}, \mathbf{a})}
            {e^{-E_\theta (\mathbf{o}, \mathbf{a})} + \sum_{j=1}^{N_{neg}}}
    \right)
$$

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

### Follow the Leader (FTL)

At reach round $t$, the agent selects the decision $x_t$ that minimizes the cumulative loss of the previous rounds:

$$
    x_t = \arg \min_{x \in \mathcal{X}} \sum_{i=1}^{t-1} l_i(x)
$$

### [Follow the Regularized Leader (FTRL)](https://arxiv.org/pdf/1912.13213.pdf)

To obtain better regret bounds, a regularizer $R: \mathcal{X} \rightarrow \mathbb{R}$ is added to the loss function:

$$
    x_t = \arg \min_{x \in \mathcal{X}} \sum_{i=1}^{t-1} l_i(x) + R(x)
$$

### Stochastic Gradient Descent (SGD)

### [Online Subgradient Descent (OGD)](https://arxiv.org/pdf/1912.13213.pdf)

For each round $t = 1,2,\cdots,T$:

1. Predict $x_t \in \mathcal{X}$.
2. Choose $z_t \in \partial l_t(x_t)$, where $\partial l_t(x_t)$ is the subgradient of $l_t(x_t)$.
3. If $\mathcal{X} = \mathbb{R}^d$, then update the prediction: $x_{t+1} = x_t - \eta z_t$
4. If $\mathcal{X} \subset \mathbb{R}^d$, then update the prediction: $x_{t+1} = \Pi_{\mathcal{X}} (x_t - \eta z_t)$, where $\Pi_{\mathcal{X}}$ is the projection operator onto $\mathcal{X}$.

### [Online Mirror Descent (OMD)](https://arxiv.org/pdf/1912.13213.pdf)

### [Mechanic (Learning Rate Tuner)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/955499a8e2860ed746717c1374224c43-Abstract-Conference.html)

## References