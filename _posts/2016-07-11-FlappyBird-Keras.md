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

lib.cnmem=0.2 means you assign 20% of the GPU memory to this programme

# What is Deep Q-Network?

Deep Q-Network is a learning algorithm developed by Google DeepMind team to play Atari games.

For those who are interested in deep reinforcement leaqrning, The following post is a must-read

[Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)


