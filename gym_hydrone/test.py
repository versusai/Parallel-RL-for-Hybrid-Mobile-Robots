import gym
import gym_hydrone
import rospy
import os

env = gym.make('hydrone_Circuit_Simple-v0', env_stage=1, observation_mode=0, continuous=True, goal_list=None)
