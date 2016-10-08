---
layout: post
comments: true
title: Using Keras and Deep Deterministic Policy Gradient to play TORCS
---

300 lines of python code to demonstrate DDPG with Keras

# Overview

This is the second blog posts on the reinforcement learning. In this project we will demonstrates how to use the Deep Deterministic Policy Gradient algorithm (DDPG) with Keras together to play TORCS (The Open Racing Car Simulator), a very interesting AI racing game and as research platform.

# Installation Dependencies:

* Python 2.7
* Keras 1.1.0
* Tensorflow r0.10
* gym_torcs         (https://github.com/ugo-nama-kun/gym_torcs)

# How to Run?

```
git clone https://github.com/yanpanlau/DDPG-Keras-Torcs.git
cd DDPG-Keras-Torcs
python ddpg.py -m "Run"
```

**If you want to train the network from beginning, delete "actormodel.h5" and "criticmodel.h5" and run ddpg.py -m "Train"**

# Motivation

As a typical child growing up in Hong Kong, I do like watching cartoon movies. One of my favourite movie is called "GPX Cyber Formula". It is an anime series about Formula racing in the future, a time when the race cars are equipped with super-intelligent AI computer system called "Cyber Systems". The AI can communate with humans interactively and it can assist drivers to race in various extreme situations. Althought we are still far away from building super-intelligent AI system yet, the latest development in the computer vision and deep learning has created an excited era for me to fullfilled my little dream -- to create a cyber system called "Asurada". 

![](/img/torcs/asurada.jpg)

# Background

In the previous blog post [Using Keras and Deep Q-Network to Play FlappyBird](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html) we demonstrate using Deep Q-Network to play FlappyBird. However, a big limitation of Deep Q-Network is that the outputs/actions are discrete while in car racing, the action like steering or acceleration are continuous. An obvious approach to adapting DQN to continuous domains is to simplu discretize the action space. However, we enouter the "curse of dimensionality", for example, if you discrtize the steering wheel from -90 to +90 degrees in 5 degree each and accelration from 0km to 300km in 5km each, your output combinations will be 36 steering states times 60 velocity space which equals to 2160 possible combinations. The situation will be worse say you want to build a robots to do brain surgery that requries fine control of actions and naive discretization will not able to achieve the required precision to do the operations.

Google Deepmind has devised a new algorithm to tackle the continuous action space problem by combining 3 techniques together 1) deterministic policy-gradient 2) actor-critic algorithm 3) DQN called [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971)

The original paper is quite hard to digest for non-machine learning expert so I will try to explain in plain english here. If you already knew the algorithm you can directly go to [Keras code session](https://yanpanlau.github.io/2016/10/13/Torcs-Keras.html)

#Policy Network
First we are going to define a **policy network** that implements our AI-driver. This network will take the state of the game (for example, velocity of the car, distance between the car and the track axis, distance between track edge and the car etc) and decide what we should do (steering left/right, hit the gas padel, hit the brake). We called it Policy-Based Reinforcement Learning because we will directly parametrise the policy

$$ \pi_\theta(s, a) = P [a | s, \theta] $$

here, s is the state , a is the action and $$\theta$$ is the model parameters of the policy network. We can think of policy is the agent's behaviour, i.e. a function to map from state to action. 

#Deterministic vs Stochastic Policy
Please note that there are 2 type of the policy:

Deterministic policy: $$a = \pi(s)$$
Stochastic policy: $$\pi(a|s) = P[a|s]$$

Why there are 2 type of policy? The deterministic policy is easy: I see a particular state input, then I take the particular action. But sometimes deterministic policy won't work, like in the example of GO, where your first state is the empty board. 

![](/img/torcs/Go_Board_9x9.png)

If you use same deterministic strategy, your network will always place the stone in a "particular" position which is a highly undesirable beheaviour. In that situation, stochastic policy is more suitable than deterministic policy.


#Policy Objective Functions
So how can I find $$\pi_\theta(s, a)$$? Actually we can use the reinformement technique to achieve that. For example, suppose you want to teach the AI to turn around a left cornor. At the beginning the AI may be simply didn't steer the wheel and hit the curb, and receive some negative reward, the neural network will adjust the model parameters $$\theta$$ such that next time it won't hit the curb. After many trail-and-error it found that "argh, if I turn the wheel a bit more left I won't hit the curb so early". In mathematics language we called it policy objective functions.

Define total discount future reward

$$R = r_1 + \gamma r_{1} + \gamma^{2} r_{2} ... + \gamma^{n} r_n$$

An initutive policy objective function will be the expectation of the total discount reward

$$L(\theta) = E[r_1 + \gamma r_2 + \gamma^{2} r_2 + ...  | \pi_\theta(s,a)]$$

$$ = E_{x~p(x|\theta)[R]}$$

where the expectations of the total reward R is calculated under some probability distribution $$p(x;\theta)$$ parameterized by some $$\theta$$

If you recall our previous blog that we have introduced the Q-function, which is maximum discounted future reward if we choose action a in state s

$$
Q(s_t, a_t) = max R_{t+1}
$$

therefore, we can write the gradient of a deterministic policy $$a=\pi(s)$$ as

$$
\frac{\partial L(\theta)}{\partial \theta} = E[\frac{\partial Q}{\partial \theta}]
$$

$$
= E[\frac{\partial Q^{\theta}(s,a)}{\partial a}\frac{\partial a}{\partial \theta}]
$$


where we applied chain rule in the last equation.

#Actor-Critic Algorithm
The Actor-Critic Algorithm is essentially a hybrid method to combine the policy gradient method and the value function method together. The policy function is known as the *actor*, while the value function is referred to as the *critic*. Essentially, the actor produces the action $$a$$ given the current state of the environment $$s$$, while the critic produces a signal to criticizes the actions made by the actor. It is natural in the human's world where the junior person do the actual work and the senior person (aka boss) always criticizes your work and make the junior person do it better. In our TORCS example we will use the Q-learning as our critic model and using policy gradient method as our actor model. The following figure explain the relationships how the Value Function/Policy Function and Actor-Critic are related together.

![](/img/torcs/actor-critic.png)

Going back to the previous equations, we can use the trick of the Deep-Q Network where we replace the Q-function as a neural network
$$Q^{\pi}(s,a) \approx Q(s,a,w)$$, where w is the weight of the neural network. Therefore, we arrived the Deep Deterministic Policy Gradient Formula:

$$
\frac{\partial L(\theta)}{\partial \theta} = \frac{\partial Q(s,a,w)}{\partial a}\frac{\partial a}{\partial \theta}
$$

#Keras Code Explanation 

##Actor Network
Let's first talk about how to build the Actor Network in Keras. Here we used 2 hidden layer with 300 and 600 units respectively. The output consist of 3 continuous action, *Steering*, which is a single unit with tanh activation function (where -1 means max left turn and +1 means max right turn). *Acceleration*, which is a single unit with sigmoid activation function (where 0 means no gas, 1 means full gas). *Brake*, another single unit with sigmoid activation function (where 0 means no brake, 1 bull brake)

```python
    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])   #input_shape=(img_channels,img_rows,img_cols))
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   #Steering from [-1, 1]
        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   #Steering from [-1, 1]
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   #Steering from [-1, 1]
        Gear = Dense(1,activation='linear',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   #Steering from [-1, 1]
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=S,output=V)
        print("We finished building the model")   
        return model, model.trainable_weights, S
```

























Once we have the policy objective function, then the problem become straightforward, I want to adjust my policy parameters $$\theta$$ to achieve more rewards. We can simply using the gradient method (aka simple calculus) to finding maximum of a function.

$$\nabla_{\theta} E_x [R] $$   Let's compute the gradient of the policy objective function

$$\nabla E_x [R] = \nabla_{\theta} \int p(x) R(x)$$     Definition of expectation


