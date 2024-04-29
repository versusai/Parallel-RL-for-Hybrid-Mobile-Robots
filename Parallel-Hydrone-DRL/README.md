<h2 align="center">Parallel Deep Reinforcement Learning for Aerial Mobile Robots</h2>

<p align="center">
  <img src="https://img.shields.io/badge/ROS-NoeticNinjemys-yellow"/>
  <img src="https://img.shields.io/badge/PyTorch-v1.9.0-blue"/>
  <img src="https://img.shields.io/badge/Torchvision-v0.8.1-blue"/>
  <img src="https://img.shields.io/badge/OpenCV-v4.4.0.46-blue"/>
  <img src="https://img.shields.io/badge/Pillow-v7.2.0-blue"/>
  <img src="https://img.shields.io/badge/Matplotlib-v3.3.3-blue"/>
  <img src="https://img.shields.io/badge/Pandas-v1.1.4-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.19.2-blue"/>
</p>
<br/>

## Objective
<p align="justify"> 
  <a>In this repository, we present a study of deep reinforcement learning techniques that uses parallel distributional actor-critic networks to navigate aerial mobile robots. Our approaches were developed taking into account only a couple of laser range findings, the relative position and angle of the robot to the target as inputs to make a robot reach the desired goal in an environment.
Based on the results gathered, it is possible to conclude that parallel distributional deep reinforcement learning’s algorithms, with continuous actions, are effective for decision-make of a aerial robotic vehicle and outperform non-parallel-distributional approaches in training time consumption and navigation capability.</a>  
</p>
  

## Setup
<p align="justify"> 
 <a>All of requirements is show in the badgets above, but if you want to install all of them, enter the repository and execute the following line of code:</a>
</p>

```shell
pip3 install -r requirements.txt
```

<p align="justify"> 
 <a>Before we can train our agent, we need to configure the config.yaml file. Some parameters seems a bit unbiguous, make sure the correct token key of comet is setted instead disable it. Another thing to do is choose an algorithm to perform your training, choose between PDDRL and PDSRL. </a>
</p>

<p align="justify"> 
 <a>With the config.yaml configured, now we can train our agent, to do this just run the following code:</a>
</p>

```shell
python3 train.py
```

