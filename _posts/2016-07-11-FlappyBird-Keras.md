---
layout: post
title: Using Keras and Deep Q-Network to Play Flappy Bird
---

# Overview
---

This project demostrated how to use Deep-Q Learning algorithm with Keras together to play Flappy Bird.

It is my first project in Machine Learning and intented to target new-comer who is interested in Reinforcement Learning.

# Installation Dependencies:
---

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

