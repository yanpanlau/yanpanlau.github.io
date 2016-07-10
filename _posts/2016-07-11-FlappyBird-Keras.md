---
layout: post
title: Using Keras and Deep Q-Network to Play Flappy Bird
---

A single 200 lines python code to demostrate DQN using Keras

# Overview

This project demostrated how to use Deep-Q Learning algorithm with Keras together to play Flappy Bird.

This is my first project in Machine Learning and intented to target new-comers who are interested in Reinforcement Learning.

# Installation Dependencies:

* Python 2.7
* Keras 1.0 
* pygame
* scikit-image

# How to Run?

CPU only

```
git clone https://github.com/yanpanlau/Keras-FlappyBird.git
cd Keras-FlappyBird
python qlearn.py
```

GPU version (Theano)

```
git clone https://github.com/yanpanlau/Keras-FlappyBird.git
cd Keras-FlappyBird
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python qlearn.py
```

Here, 'lib.cnmem=0.2' means you assign 20% of the GPU memory to this programme

# What is Deep Q-Network?

Deep Q-Network is a learning algorithm developed by Google DeepMind team to play Atari games.

For those who are interested in deep reinforcement learning, The following post is a must-read

[Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)


# Code Explaination (in details)

Let's go though the example in 'qlearn.py', line by line. If you familiar with Keras and DQN, please skip this session

Here is what the code does: 

1. I receive the Game Screen Input (Image) in a form of pixel array
2. I do some image pre-processing 
3. The processed image will be feed into a neural network (Convolution Neural Network), and the network will decide the best action (Flap or not Flap)
4. The neural network will be trained millions of times, via a algorithm called Q-learning, to maximize the future expected rewards.

First of all, the FlappyBird is already written in Python via Pygame, here is the code snippet to access the FlappyBird API

```python
import wrapped_flappy_bird as game
x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
```
So the idea is very simple, the input is a_t (0 represent not flapped, 1 represent flapped), then the API will give you the next frame 'x_t1_colored', the reward (0.1 is not die, -100 if die) and 'terminal' indicates whether GAMEOVER or not.

Now, in order to make the code run faster and converge faster, it is vital to do some image processing. Here are the key elements

1. I first convert the color image into grayscale
2. I corp down the image size into standard 80x80 pixel
3. I stack 4 frames together before I feed into neural network. 

So why do I need to stack 4 frames? This is one way to provide "velocity" information of the bird.

```python
x_t1 = skimage.color.rgb2gray(x_t1_colored)
x_t1 = skimage.transform.resize(x_t1,(80,80))
x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)
```
So x_t1 is a single frame with shape (1x1x80x80) and s_t1 is the stacked frame with shape (1x4x80x80). You might asked, why the input dimension is (1x4x80x80) but not (4x80x80)? Well, it was an requirement in Keras.

Now, We can input the pre-processed screen into the neural network, which is a convolution neural network:

```python
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
   
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model
```

The exact architecture is as follows : The input to the neural network consistes of an 4x80x80 images. The first hidden layer convolves 32 filters of 8 x 8 with stride 4 and applies ReLU activation function. The 2nd layer convolves a 64 filters of 4 x 4 with stride 2 and applies ReLU activation function. The 3rd layer convolves a 64 filters of 3 x 3 with stride 1 and applies ReLU activation function. The final hidden layer is fully-connected consisted of 512 rectifier units. The output layer is a fully-connected linear layer with a single output for each valid action.

So wait, what is convolution? The easiest way to understand a convolution is by thinking of it as a sliding window function apply to a matrix. The following gif file should help to understand.

You might ask what's the purpose the convolution? It actually helps computer to learn higher features like edges and shapes. See the example below:



Keras makes it very easy to build convolution neural network. However, there are few things I would like to highlight here

1. It is important to choose a right initialization method. Here I choose normal distribution with $sigma=0.01$

```
init=lambda shape, name: normal(shape, scale=0.01, name=name)
```

2. The ordering of the dimension is important, the default setting is 4x80x80 (Theano setting) so if you input as 80x80x4 (Tensorflow setting) then you are in trouble. If your input dimension is 80x80x4 you need to set $dim_ordering = tf$ (tf means tensorflow)

3. In Keras, the subsample=(2,2) means you downsample the image size from (80x80) to (40x40). In literature it is often called "stride"

4. We have used an adaptive learning algorithm called ADAM to do the optimization. The learning rate is 1-e6. 

Finally, we can using the Q-learning algorithm to train the neural network.

So, what is Q-learning? In Q-learning what matters is a Q function : Q(s, a) representing the maximum discounted future reward when we perform action a in a state s. Q(s, a) gives you an estimation of how good to choose an action a in a state s. Now, you might ask 1) Why Q-function is useful? 2) How can I get the Q-function?

Suppose you are playing a difficult RPG game and you don't know how to play it well. If you have bought a Strategy guide, which is an instruction books that contain hints or complete solutions to a specific video games. Then playing video game is easy. You just follow the guidience of the strategy book. Here, Q-function is like a strategy guide. Suppose you are in state and you need to decide whether you take action a or b. If you have this magical Q-function, the answer will become really siomple -- pick the action with highest Q-value!

$$ \pi(s) = argmax Q(s,a) $$

Here, $\pi$ represents the policy, you will often see that in the literature.

Now, how do we get the Q-function then? That's where is Q-learning coming from. Here I quickly derive it:

Define discounted future reward

$$
R_t = r_t + \gamma r_{t+1} + \gamma^{2} r_{t+2} ... + \gamma^{n-t} r_n
$$

which, can be also written as

$$
R_t = r_t + \gamma * R_{t+1}
$$

As we discuss above, the definition of Q-function is below

$$
Q(s_t, a_t) = max R_{t+1}
$$

therefore, we can re-write the Q-fuction as below

$$
Q(s, a) = r + \gamma * max_{a^'} Q(s^', a^')
$$

In plain English, it means maximum future reward for this state and action is the immediate reward plus maximum future reward for the next state.

Now, we could use iterative method to solve the Q-function. Given a transition $<s, a, r, s^'>$ , we are going to convert this episode into training set for the network. i.e. We want $r + \gamma max_a Q (s,a)$ to be equal to $Q(s,a)$

we can define a loss function below

$$
L = {r + max Q(s, a) - Q (s, a)}^2
$$

You can think of finding a Q-value is regression task now. Given a transition $<s, a, r, s^'>$, how can I optimized my Q-function such that it return smallest simple squared error loss? If L goes to zero, it means, the Q-function is converged into the optimal value, which is our "strategy book" we need.

Now, you might ask, hey, where is the role of the neural network? Here is where the "DEEP Q-Learning" coming. You recall that $Q(s,a)$, is a stategy book, which contains millions or trillions of states and actions, if you implemented as a table. The idea of the DQN is that I use the neural network to **COMPRESS** this Q-table, using some parameters $\theta$ [In neural network we called it weight]. So instead of handle a large table, I just need to worry the weights of the neural network going forward. But hopefully I smartly tune the weight parameters, I can find the optimal Q-function.

$$
Q(s,a) = f_{\theta}(s)
$$

where $f$ is our neural network with input $s$ and parameters $\theta$

Here is the code below to demostrate how it works
```python
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1
``` 

If you examine the code above, there is a comment called "Experience Replay". Let me explain what it does: It was found that approximation of Q-value using non-linear functions like neural network is not very stable. The most important trick to solve this problem is called **experience replay**. During the gameplay all the episode $<s, a, r, s^'>$ are stored in replay memory D. [I use Python function deque() to store it]. When training the network, random minibatch from the replay memory are used instead of most recent transition, which will greatly improve the stability.

That's it. I hope this small tutorial will help you to understand how DQN works. 


# FAQ

## My training is very slow

You might need a GPU to accelerate the calculation. I used a TITAN X and train for at least 1 million frame to make it work

# Future works and thinking

1. Current DQN depends on large experience replay. Is it possible to replace it or even remove it?
2. How can one decide optimal Convolution Neural Network?
3. Training is very slow, how to speed it up/converge faster?
4. What does the Neural Network actually learnt? Is the knowledge transferable?

I believe the questions still not resolved and it's an active research area in Machine Learning.


# Reference

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

#Disclaimer

This work is highly based on the following repos:

1. https://github.com/yenchenlin/DeepLearningFlappyBird

2. http://edersantana.github.io/articles/keras_rl/

#Acknowledgement

I must thank to @hardmaru to encourage me to write this blog. I also thank to @fchollet to help me on the weight initialization in Keras and @edersantana his post on Keras and reinforcement learning which really help me to understand it.





