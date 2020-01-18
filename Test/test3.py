import gym
import numpy as np
import random

env = gym.make('CartPole-v0')
env.reset()
print(env.step(1))
print(env.step(1))
print(env.reset())
a, b, c, d = env.reset()
print(a, b, c, d)
print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
print(env.action_space.sample())
print(env.observation_space.sample())
print(env.action_space.n)
print(env.reset())
env.close()
