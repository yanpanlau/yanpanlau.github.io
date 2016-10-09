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


##Why TORCS
I think it is important to study TORCS because:

* It's looks cool, it's really cool to see the AI can learn how to drive in unsupervised learning
* You can visualize how the neural network learn how to drive, not just the end result
* It is easy to visualize when the neural network stuck in local minimun
* It can help to understand machine learning technique in automation driving
* It can see if computer can find a better policy than human drivers

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

$$= E_{x~p(x|\theta)[R]}$$


where the expectations of the total reward R is calculated under some probability distribution $$p(x;\theta)$$ parameterized by some $$\theta$$

If you recall our previous blog that we have introduced the Q-function, which is maximum discounted future reward if we choose action a in state s

$$Q(s_t, a_t) = max R_{t+1}$$

where in the continuous case (SARSA) we can written as

$$Q(s_t, a_t) = R_{t+1}$$

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

Follow the previous DQN blog post, we could use an iterative method to solve for the Q-function, where we can setup the Loss function

$$
L = [r + \gamma Q (s^{'},a^{'}) - Q(s,a)]^{2}
$$

Once the Q-function converged to the optimal value, the Q-value can be used to estimate the values of the current actor policy.

#Keras Code Explanation 

##Actor Network
Let's first talk about how to build the Actor Network in Keras. Here we used 2 hidden layer with 300 and 600 hidden units respectively. The output consist of 3 continuous action, *Steering*, which is a single unit with tanh activation function (where -1 means max left turn and +1 means max right turn). *Acceleration*, which is a single unit with sigmoid activation function (where 0 means no gas, 1 means full gas). *Brake*, another single unit with sigmoid activation function (where 0 means no brake, 1 bull brake)

{% highlight python %}

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size]) 
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh')(h1)   #Steering from [-1, 1]
        Acceleration = Dense(1,activation='sigmoid')(h1)   #Steering from [-1, 1]
        Brake = Dense(1,activation='sigmoid')(h1)   #Steering from [-1, 1]
        Gear = Dense(1,activation='linear')(h1)   #Steering from [-1, 1]
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=S,output=V)
        print("We finished building the model")   
        return model, model.trainable_weights, S

{% highlight ruby %}

We have used Keras function called "Merge" to combine 3 outputs togther. Smart reader may asked why not using traditional Dense function like this

```python

V = Dense(3,activation='tanh')(h1)   #Directly output 3 values for Steering, Acceleration, Brake

```

There is a reason for that. First using 3 different Dense() function allows each continous action have different activation function, for example, using tanh() for acceleration doesn't make sense as tanh are [-1,1] while the accelration is in the range [0,1]


##Critic Network
The construction of the Critic Network is very similar to the Deep-Q Network in the previous post. The only difference is that we used 2 hidden layer with 300 and 600 hidden units. Also, the critic network takes both the states and the action as inputs. According to the DDPG paper, the actions were not included until the 2nd hidden layer of Q-network. Here we used the Keras function "Merge" to merge the action and the hidden layer together

```python

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim],name='action2')    #action = tf.placeholder(tf.float32, shape=[None, action_dim])
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S) #h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) #     tf.matmul(action, W2_action) + b2)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)#     tf.matmul(h1, W2)
        h2 = merge([h1,a1],mode='sum')    #Both have HIDDEN2_UNITS       #tf.matmul(h1, W2) + tf.matmul(action, W2_action) + b2
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)   #In Q learning, if you do not specify, it is by default linear
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("We finished building the model")
        return model, A, S 

```

##Target Network 
It is a well known fact that directly implementing Q learning with neural networks proved to be unstable in many environment including TORCS. Deepmind team came up the solution to the problem is to use a target network, where we created a copy of the actor and critic networks respectively, that are used for calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks: 

$$\theta^{'} \leftarrow  \tau \theta + (1 - \tau) \theta^{'}$$

where $$\tau \ll 1$$. This means that the target values are constrained to change slowly, greatly improving the stability of learning.

It is extremely easy to implement target networks in Keras:

```python

    def target_train(self):
        #self.sess.run(self.target_update)  
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

```

#Main Code
After we finished the network setup, Let's go though the example in **ddpg.py**, our main code

The code simply does the following:

1. The code receives the sensor input in the form of array
2. The senor input will be fed into our Neural Network, and the network will output 3 real numbers (mangitute of the steering, acceleration and brake)
3. The network will be trained many times, via the Deep Deterministic Policy Gradient, to maximize the future expected reward.


##Sensor Input
In the TORCS there are 18 different types of sensor input, the details can be found here [Simulated Car Racing Championship : Competition Software Manual](https://arxiv.org/abs/1304.1672). So which sensor input should we used? After some trial-and-error, I found the following inputs are useful:

| Name          | Range (units) | Description  |
| :------------ |:-----       | :----------  |
| ob.angle      | [-$$\pi$$,+$$\pi$$]       | Angle between the car direction and the direction of the track axis |
| ob.track      | (0, 200)(meters)      |   Vector of 19 range finder sensors: each sensors returns the distance between the track edge and the car within a range of 200 meters |
| ob.trackPos | (-$$\infty$$,+$$\infty$$)      |   Distance between the car and the track axis. The value is normalized w.r.t. to the track width: it is 0 when car is on the axis, values greater than 1 or -1 means the car is outside of the track.
| ob.speedX | (-$$\infty$$,+$$\infty$$)(km/h) | Speed of the car along the longitudinal axis of the car (good velocity) |  
| ob.speedY | (-$$\infty$$,+$$\infty$$)(km/h) | Speed of the car along the transverse axis of the car |  
| ob.wheelSpinVel |  (0,+$$\infty$$)(rad/s)   | Vector of 4 sensors representing the rotation speed of wheels |
| ob.rpm |  (0,+$$\infty$$)(rpm)              | Number of rotation per minute of the car engine |

Please do note that we have normalized some of those value before feed into the neural network and in gym_torcs some sensor inputs are not exposed. Advanced user need to amend gym_torcs.py to change the parameters. [checkout the function make_observaton()]

##Policy Selection
Now we can use the inputs above to feed into the neural network. The code is actually very simple:

```python

    for j in range(max_steps):
        a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
        ob, r_t, done, info = env.step(a_t[0])

```

However, we immediately run into the two issues, First, how do we decide the reward? Second, how do we do exploration in the continuous action space? 

###Design of the rewards
In the original paper, they used the reward function which is equal to the velocity of the car projected along the track direction. 
$$ V_x cos(\theta)$$. The following picture show explain clearly.

![](/img/torcs/velocity.png)

However, I found that the reward function is not very stable and a lot of policy failed to converge, as reported in the original paper. 

> On both low-dimensional and form pixels, some replicas were able to learn reasonable policies that are able to complete a circuit arond the track though other replicas failed to learn a sensible policy


I believe the reason is that in the original policy the AI will try to accelerate the gas padel as hard as it can and it the hit the edge and the episode terminated very quickly. Therefore, the netural network stuck in a very poor local minimun. The new proposed reward function is below:

$$R_t = V_x cos(\theta) - V_y sin(\theta) - V_x \mid trackPos \mid$$

In plain english, we want to maximum logitudinal velocity (first term), minimize transverse velocity (second term), and we also penalitze the AI if it constantly drive very off center of the track (third term)

I found the new reward function greatly improves the stability and the learning time of TORCS.

###Design of the exloration algorithm
Another issue is how to design a right exploration algorithm in continuous domain. In the previous blog post we used $$\epsilon$$ greedy policy where the agent to try a random action some percentage of the time. However, this approch does not work very well in TORCS because we have 3 actions [steering,acceleration,brake]. If I just randomly choose the action from uniform random distribution we will generate some un-interesting action pair [eg, when the value of the brake is greater than the value of acceleration], where the car simply not moving. Therefore, we add the noise using Ornstein-Uhlenbeck process to generate the noise.

###Ornstein-Uhlenbeck process
What is Ornstein-Uhlenbeck process? In simple english it is simply a stochastic process which have mean-reverting properties. 

$$dx_t = \theta (\mu - x_t)dt + \sigma dW_t$$

here, $$\theta$$ means the how "fast" the varaible reverts towards to the mean. $$\mu$$ represents the equilibrum or mean value. $$\sigma$$ is the degree of volatility of the process. Interestingly, Ornstein-Uhlenbeck process is very common approach to model interest rate, FX and commodity prices stochastically. (And a very common interview questions in finance quant interview). The following table shows the suggested values that I used in the code.


| Action        | $$\theta$$    | $$\mu$$  |  $$\sigma$$ | 
| :------------ |:------------- | :-----   |   :-----    |
| steering      | 0.6           |   0.0    |    0.30     |
| acceleration  | 1.0           |   0.5    |    0.10     |
| brake         | 1.0           |  -0.1    |    0.05     |

Basically, the most important parameters is the $$\mu$$ of the acceleration, where you want the car have some initial velocity and don't stuck in a local minimun where the car keep pressing the brake and never hit the gas padel. Readers are feel free to change the parameters and see how the AI performs in various combination. The code of the Ornstein-Uhlenbeck process is saved under **OU.py**

My research finding is that the AI can learn a reasonble policy on the simple track very quickly if using the above exploration policy and revised reward function, like around ~50 esipode.

##Experience Replay
Similar to the FlappyBird case, we also used the Experience Replay to saved down all the episode $$ (s, a, r, s^{'}) $$ in a replay memory. When training the network, random mini-batches from the replay memory are used instead of most the recent transition, which will greatly improve the stability. The following code snippet shows how it is done.

```python

        buff.add(s_t, a_t[0], r_t, s_t1, done)
        # sample a random minibatch of N transitions (si, ai, ri, si+1) from replay buffer
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])    #Still using tf
       
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]

```

Please note that when we calculated the target_q_values we do use the output of **target_model** instead of the **model** instead. The used of the slow-varing target_model will reduced the oscillations of the Q-value estimation, which greatly improve the stability of the learning.

##Training
The actual training of the neural network is very simple, only contains 6 lines of code:

```python

        loss += critic.model.train_on_batch([states,actions], y_t) #Input s_i, a_i, mininize the loss function
        a_for_grad = actor.model.predict(states)
        grads = critic.gradients(states, a_for_grad)
        actor.train(states, grads)
        actor.target_train()
        critic.target_train()

```
In plain english, we first update the critic by minimizing the loss 

$$L = \frac{1}{N} \displaystyle\sum_{i} (y_i - Q(s_i,a_i | \theta^{Q})) $$

Then the actor policy is updated using the sampled policy gradient

$$\nabla_\theta J = \frac{\partial Q^{\theta}(s,a)}{\partial a}\frac{\partial a}{\partial \theta}$$

but $$a=\mu(s \mid \theta)$$

therefore, it can written as

$$\nabla_\theta J = \frac{\partial Q^{\theta}(s,a)}{\partial a}\frac{\partial \mu(s|\theta)}{\partial \theta}$$

The last 2 lines of the code is to used to update the target network


$$\theta^{Q^{'}} \leftarrow  \tau \theta^{Q} + (1 - \tau) \theta^{Q^{'}}$$

$$\theta^{\mu^{'}} \leftarrow  \tau \theta^{\mu} + (1 - \tau) \theta^{\mu^{'}}$$


#Results
I trained the neural network with 2000 espoide, and the noise process allow to decay linearly in 100000 frames. (i.e. no more exploriation is applied after 100k frame). I choose the track **Aalborg** for my training dataset and I will validate my neural network by allow the AI to drive on another track **Alpine 1**. It is important to test the AI agents in other tracks in order to prove that the AI didn't simply "memorize the track", or aka overfitting.

The first video shows the results in the **Aalborg** track





#Misc things you need to know

1) To try different track, you need to type **sudo torcs** --> Race --> Practice --> Configure Race

2) Installation of the TORCS requires openCV. I have some hard time to install correctly as it crashed my NVIDIA Titan-X driver. I strongly suggest you download a copy of the NVIDIA driver in the local drive first. In case if your video driver is crashed you can restore your video driver by install the video card driver in the text mode

3) To turn off the engine sound during training you can type **sudo torcs** --> Options --> Sound --> Disable sound



























[^Comment]: Once we have the policy objective function, then the problem become straightforward, I want to adjust my policy parameters $$\theta$$ to achieve more rewards. We can simply using the gradient method (aka simple calculus) to finding maximum of a function.
[^Comment]: $$\nabla_{\theta} E_x [R] $$   Let's compute the gradient of the policy objective function
[^Comment]: $$\nabla E_x [R] = \nabla_{\theta} \int p(x) R(x)$$     Definition of expectation


