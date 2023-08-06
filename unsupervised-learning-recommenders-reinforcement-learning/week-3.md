# Week 3 - Reinforcement learning

## What is Reinforcement Leaning

For the jobs like autonomous flying helicopter, we can't use supervised learning to train the model. Because we can't get the data set of the helicopter flying. So we need to use reinforcement learning to train the model.

In reinforcement learning, we don't have the data set. We only have the reward function. We can use the reward function to train the model.

In reinforcement learning, instead of telling the model what to do, we tell the model what is good and what is bad. Then the model will learn from the reward function.

Applications:

- Controlling robots
- Factory optimization
- Financial (stock) trading
- Playing games (including video games)

## Mars rover example

<img src="assets/img-11.jpg" height="300"/>

A robot is on Mars and it has to get to the goal. The robot can move forward or backward. In each step, the robot gets a reward. The robot has to get to the goal with the maximum reward.

The rewarded steps at the two ends, are called _terminal state_ because the robot can't move forward or backward from there.

At every time-step, the robot gets a reward. The robot is at state $s$, and it gets to choose an action $a$, and it gets a reward $R(s)$ and it gets to a new state $s'$: $(s, a, R(s), s')$

For example when robot is at state $4$, it take the action of going to the left, and gets $0$ reward, and then it will be at state $3$: $(4, \leftarrow, 0, 3)$

## The Return in reinforcement learning

The output of reward function may vary and we may have to take some steps without gaining any reward to reach a good reward at last, but what defines if it worth it or not?

Here we define _Discount Factor_ ($\gamma$) which is a little bit less than $1$ and we use it to define the return of the reward function.

For example with $\gamma = 0.9$, the total reward is:

$$
R = \sum_{k=0}^{\infty} \gamma^k r_{k+1}
$$

In financial application, the discount factor has a natural interpretation as the interest rate or the time value of money.

To summarize, the return is sum of the rewards that the system gets, weighted by the discount factor.

## Making decisions: Policies in reinforcement learning

A policy is a function $\pi(s) = a$ mapping from states to actions, that tells you what action a to take in a given state $s$.

Our goal reinforcement learning is to find a policy $\pi$ that tells you what action ($a = \pi(s)$) to take in every state ($s$) so as to maximize the return.

## Review of key concepts

Markov Decision Process (MDP): This term refers to that the future only depends on the current state, not the past states. In other words, future only depends on what you are now, not how you got here.
