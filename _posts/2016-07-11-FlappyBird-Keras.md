---
layout: post
title: Using Keras and Deep Q-Network to Play Flappy Bird
---

# Overview

This project demostrated how to use Deep-Q Learning algorithm with Keras together to play Flappy Bird.

It is my first project in Machine Learning and intented to target new-comer who is interested in Reinforcement Learning.

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

The exact architecture is as follows : The input to the neural network consistes of an 4x80x80 images. The first hidden layer convolves 32 filters of 8 x 8 with stride 4 and applies ReLU activation function. The 2nd layer convolves a 64 filters of 4 x 4 with stride 2 and applies ReLU activation function. The 3rd layer convolves a 64 filters of 3 x 3 with stride 1 and applies ReLU activation function. The final hidden layer is fully-connected consisted of 512 rectifier unites. The output layer is a fully-connected linear layer with a single output for each valid action.

Keras makes it very easy to build convolution neural network. However, there are few things I would like to highlight here

1. It is important to choose a right initialization method. Here I choose normal distribution with $sigma=0.01$

```
init=lambda shape, name: normal(shape, scale=0.01, name=name)
```

2. The ordering of the dimension is important, the default setting is 4x80x80 (Theano setting) so if you input as 80x80x4 (Tensorflow setting) then you are in trouble. If your input dimension is 80x80x4 you need to set $dim_ordering = tf$ (tf means tensorflow)


